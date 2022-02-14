#!/usr/bin/python

import hashlib

if __name__ == "__main__":
    md5 = hashlib.md5()
    md5.update('This is a sentence.'.encode('utf-8'))
    md5.update('This is a second sentence.'.encode('utf-8'))
    print('1.不出意外，这个将是“乱码”：', md5.digest())
    print('2.MD5:', md5.hexdigest())

    md5 = hashlib.md5()
    md5.update('3.This is a sentence.This is a second sentence.'.encode('utf-8'))
    print('4.MD5:', md5.hexdigest())
    print(md5.digest_size, md5.block_size)
    print('------------------')

    sha1 = hashlib.sha1()
    sha1.update('5.This is a sentence.'.encode('utf-8'))
    sha1.update('6.This is a second sentence.'.encode('utf-8'))
    print('7.不出意外，这个将是“乱码”：', sha1.digest())
    print('8.SHA1:', sha1.hexdigest())

    sha1 = hashlib.sha1()
    sha1.update('9.This is a sentence.This is a second sentence.'.encode('utf-8'))
    print('10.SHA1:', sha1.hexdigest())
    print(sha1.digest_size, sha1.block_size)
    print('=====================')

    md5 = hashlib.new('md5', 'This is a sentence.This is a second sentence.'.encode('utf-8'))
    print(md5.hexdigest())
    sha1 = hashlib.new('sha1', 'This is a sentence.This is a second sentence.'.encode('utf-8'))
    print(sha1.hexdigest())

    print(hashlib.algorithms_available)