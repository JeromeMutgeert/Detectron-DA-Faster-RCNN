#LANGUAGE python3
# Python 3 required. This is a seperate process.

import aiohttp
import asyncio
import async_timeout
import os
import numpy as np
import time

import cv2

# Settings:
Buffer = 100
BackBuffer = 20
Max_Query = 20
Timeout = 30


ids_file = "ids.txt"
shuffling = False

cacheLoc = "/media/jerome/DATA/Study_d/ThesisD/TargetData/"


url_base = "https://test.yisual.com/images/media/download/picturethis/"
headers = {"api-key": "ccea03e0e3a08c428870393376e5cf7b7be7a55c", "api-secret": os.environ["SECRET"]}


# dummy_im_id = "5461e5219da59bde29aed195"
# dummy_url = url_base + dummy_im_id

# counter txt's interface:
def update_fetched(fetched):
    with open("fetched_temp.txt",'w') as f:
        f.write(str(fetched))
        f.flush()
        os.fsync(f.fileno())
    # atomic:
    os.rename("fetched_temp.txt","fetched.txt")
        
def get_read():
    global read
    with open("read.txt",'r') as f:
        numstr = f.read()
    read = int(numstr)
    return read

def to_filename(im_id):
    return "target_{}.jpg".format(im_id)

def append_log(msg):
    with open("log.txt",'a') as f:
        f.write(str(time.time()) + ' :\t')
        f.write(str(msg) + '\n')


async def download_coroutine(session, im_id, im_num):
    # im_id = "5c59addcb71ee102f1e439ba"
    cache = cacheLoc + im_id + '.jpg'
    filename = to_filename(im_num)

    
        # return

    url = url_base + im_id
    im = None
    problematic = False
    this_timeout = Timeout
    while type(im) == type(None):# and (not problematic):
        try:
            if os.path.exists(cache):
                # copy from cache:
                os.system('cp {} ./{}'.format(cache,filename))
                res = True

            else:
                with async_timeout.timeout(this_timeout):
                    async with session.get(url,headers=headers) as response:
                        with open(filename, 'wb') as f_handle:
                            while True:
                                chunk = await response.content.read(1024*128)
                                if not chunk:
                                    # print('done')
                                    break
                                f_handle.write(chunk)
                            f_handle.flush()
                            os.fsync(f_handle.fileno())
                        res = await response.release()

            # Verify if download was succesfull:
            im = cv2.imread(filename)
            if type(im) == type(None):
                problematic = True
                append_log("{} {} Incorrect download.".format(im_num,im_id))
                print("{} {} Incorrect download.".format(im_num,im_id))

        except:
            problematic = True
            this_timeout += 10
            append_log("Downloading timed out, retrying {} {}".format(im_num,im_id))
            print("Downloading timed out, retrying {} {}".format(im_num,im_id))

    if problematic:
        append_log("Succeeded! {} {}".format(im_num,im_id))

    # Finally:
    if os.path.exists(cacheLoc):
        os.system('cp {} {}'.format(filename,cache))

    return res

 
async def get_batch(loop,im_ids,im_nums):
    async with aiohttp.ClientSession(loop=loop) as session:
        tasks = [download_coroutine(session, im_id, im_num) for im_id,im_num in zip(im_ids,im_nums)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":

    # init gobals
    present = []

    fetched = 0
    read = 0
    removed = 0

    idle_count = 0

    # init/reset protocol files:
    update_fetched(0)
    with open("read.txt",'w') as f:
        f.write('0')
        f.flush()
        os.fsync(f.fileno())

    ids = []
    with open(ids_file,'r') as f:
        ids = [i.strip() for i in f.readlines()]
        
    def shuffle(ids, epoch):
        if shuffling:
            np.random.shuffle(ids)
        filename = "ids_ep{}.txt".format(epoch)
        with open(filename,'w') as f:
            f.write('\n'.join(ids))
            f.flush()
        os.system('cp {} ids_current.txt'.format(filename))
    
    def id_generator():
        i = 0
        epoch = 0
        shuffle(ids,epoch)
        while True:
            yield ids[i]
            i += 1
            if i == len(ids):
                i = 0
                epoch += 1
                shuffle(ids,epoch)
                print("Loaded epoch {}".format(epoch))

    id_gen = id_generator()

    append_log("Starting")
                
    while True:

        # update read
        read = get_read()

        # print(fetched,read,removed)

        # refill:
        if (fetched - read) < Buffer:
            # TODO: determine next imgs:
            load_N = read + Buffer - fetched
            load_N = min(load_N,Max_Query)
            
            im_nums = [str(i) for i in range(fetched,fetched+load_N)]
            im_ids = [next(id_gen) for _ in range(fetched,fetched+load_N)]
            
            loop = asyncio.get_event_loop()
            loop.run_until_complete(get_batch(loop,im_ids,im_nums))

            # done fetching
            fetched += load_N
            present.extend(im_nums)
            
            # broadcast
            update_fetched(fetched)

            idle_count = 0
            
        else:
            # we're all set

            # Check for exitting:
            stop = False
            idle_count += 1
            if idle_count > 1000: # about 3 mins idle
                append_log("Idle time-out. Exiting.")
                stop = True
            if (fetched - read) > Buffer: # read.txt has decreased:
                append_log("Read.txt has decreased. Exiting.")
                stop = True
            if stop:
                for im_num in present:
                    os.remove(to_filename(im_num))
                exit()

            # sleep a bit to avoid spinning.
            time.sleep(.2)

        # remove
        while removed < (read - BackBuffer):
            try:
                im_num = present[0]
            except:
                append_log("Non-existing file reported as read. Exiting.")
                exit()
            present = present[1:]

            try:
                os.remove(to_filename(im_num))
            except:
                append_log("While removing: File not found: {}".format(to_filename(im_num)))
                print("While removing: File not found: {}".format(to_filename(im_num)))

            removed += 1