import abc
from abc import ABCMeta

class StudentDAO:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getAllStudents(self):
        """ Returns a list of students with respective check_ins"""
        pass
