# mainclass.py
from subclass import SubClass

class MainClass(SubClass):
    def __init__(self):
        super().__init__()
        print("MainClass initialized")
    
    def main_method(self):
        print("This is a method from MainClass")
    
    def use_sub_method(self):
        self.sub_method()  # 调用子类的方法

if __name__ == "__main__":
    obj = MainClass()
    obj.main_method()
    obj.use_sub_method()
