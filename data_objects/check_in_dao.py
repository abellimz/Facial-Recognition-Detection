import abc

class CheckInDAO(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def getAllCheckIns(self):
        """ Returns a dictionary of studentName: [CheckIn]"""
        pass
