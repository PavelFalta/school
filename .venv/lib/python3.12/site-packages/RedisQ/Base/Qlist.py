import pickle
from io import BytesIO
from redis import Redis


class BaseList(object):
    """if you want to create a new list just extend this class"""
    conn = None
    list_name = __name__

    # job_hash传入一个类而不是一个实例，这是为了尽量简化传参过程
    def __init__(self, connection=None, list_name=None, job_hash=None):
        # 断言部分如果通过就没事，如果不通过则抛出AssertionError
        assert connection is not None, "the redis connection is required"
        assert job_hash is not None, "the job hash is required"
        self.conn = connection
        self.job_hash = job_hash(self.conn)
        if list_name:
            # 180124LLR,使用不同的对象初始化后，一定能得到不同的队列名称，因为先加载子类变量
            # 如果有传入list_name，那么不会使用类中默认的list_name
            self.list_name = list_name

    def __iter__(self):
        return self

    def __next__(self):
        if len(self) > 0:
            ret = self.pop()
            return ret
        else:
            raise StopIteration()

    # 重写对象中的获得长度方法，能够得到队列中的对象数目
    def __len__(self):
        return self.conn.llen(self.list_name)

    def queue(self, job):
        """queues a job into this list"""
        # 业务逻辑：先创建一个任务，再放入队列中，最后更改任务状态
        # 1.在远程任务字典中创建这个任务
        # 2.更改任务的状态
        job.status = "queue into {0}".format(self.list_name)
        self.job_hash.set(job)
        # 3.把这个任务的id从左边推入队列
        self.conn.lpush(self.list_name, str(job.id))

    def pop_to(self, another_list):
        """move the last job in this list into another, return None if no job is found"""
        # 1.尝试将任务队列中最右边的任务抢到自己的工作队列中
        job_id = self.conn.rpoplpush(self.list_name, another_list.list_name)
        # 2.如果抢到了，就返回这个任务，如果抢不到（抢的时候正好没有了）就返回无
        if not job_id:
            return
        job = self.job_hash.fetch(job_id)
        if not job:
            return
        job.status = 'move form {this_list} into {list_name}'.format(
            this_list=self.list_name,
            list_name=another_list.list_name)
        self.job_hash.set(job)
        return job

    def pop(self):
        # 1.尝试将任务队列中最右边的任务pop出列
        job_id = self.conn.rpop(self.list_name)
        # 2.如果能够pop出列
        if job_id:
            job = self.job_hash.fetch(job_id)
            self.job_hash.remove(job_id)
            return job

    def pull(self):
        # 1.尝试将任务队列中最左边的任务pull出列
        job_id = self.conn.lpop(self.list_name)
        # 2.如果能够pull出列
        if job_id:
            job = self.job_hash.fetch(job_id)
            self.job_hash.remove(job_id)
            return job

    # 1.不能做到pull+rpush的事务性。2.尽量保证任务pop之后是可以完成的，不要用到这个方法
    def re_queue(self, another_list):
        job = self.pull()
        another_list.rpush(job)

    def rpush(self, job):
        if job:
            self.conn.rpush(self.list_name, str(job.id))

    def set(self, index, job):
        """modify list_name-index's job"""
        self.job_hash.set(job)
        self.conn.lset(self.list_name, index, job.id)

    def get(self, index):
        """returns the job from this list at the index"""
        # 只取信息，不取实体
        job_id = self.conn.lindex(self.list_name, index)
        print(job_id)
        if job_id:
            return self.job_hash.fetch(job_id)


# 签名的作用可能是为了不让下一个读取
class HS(object):
    """the hash of holding sign workers worker:working list"""
    hash_name = 'sign_hash'
    conn = None

    def __init__(self, connection=None, hash_name=None):
        assert connection is not None, "the redis connection is required"
        self.conn = connection
        if hash_name:
            self.hash_name = hash_name

    def get_member(self):
        return self.conn.hgetall(self.hash_name)

    def sign_in(self, worker, list):
        self.conn.hset(self.hash_name, worker.id, list.list_name)

    def sign_out(self, worker):
        self.conn.hget(self.hash_name, worker.id)


# 180124LLR，目前的设计，redis中只留一个名叫job_hash的哈希表，任务入则加，去则删
class HJ(object):
    """the hash of holding job_id:job"""
    hash_name = 'job_hash'
    conn = None

    def __init__(self, connection=None, hash_name=None):
        assert connection is not None, "the redis connection is required"
        self.conn = connection
        if hash_name:
            self.hash_name = hash_name

    def keys(self):
        return self.conn.hgetall(self.hash_name)

    def set(self, job):
        _job = pickle.dumps(job)
        # 此方法能够实现将任务键值对存入或修改job_hash这个远程字典
        self.conn.hset(self.hash_name, job.id, _job)

    def fetch(self, id):
        job = self.conn.hget(self.hash_name, id)
        if job:
            return pickle.loads(job)

    def remove(self, id):
        pass


class LQ(BaseList):
    """the list of holding queue jobs"""
    list_name = 'queue_list'


class LW(BaseList):
    """the list of holding working jobs"""
    list_name = 'work_list'


class LD(BaseList):
    """the list of holding done jobs"""
    list_name = 'done_list'


class LF(BaseList):
    """the list of holding failed jobs"""
    list_name = 'failed_list'
