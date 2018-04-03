import abc

class StudentDAO(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getAllStudents(self):
        """ Returns a list of students with respective check_ins"""
        pass
