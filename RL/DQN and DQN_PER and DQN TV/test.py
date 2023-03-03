import threading
import time
import random
def func(num):
    r= random.random()
    print("hello",num,r)
    time.sleep(r)

if __name__ =="__main__":
    t1 = threading.Thread(target=func,args=(1,))
    t1.start()
    t2 = threading.Thread(target=func,args=(2,))
    t2.start()
    t3 = threading.Thread(target=func,args=(3,))
    t3.start()
    t1.join()
    t2.join()
    t3.join()