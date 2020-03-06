from pymemcache.client.base import Client


class CacheClient(Client):
    def __init__(self):
        super(CacheClient, self).__init__(('localhost', 11211))

