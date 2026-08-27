from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text="""
class Student:
    # Constructor - runs when a new object is created
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    # Method - defines behavior of the class
    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Grade: {self.grade}")

    def is_adult(self):
        return self.age >= 18


# Creating objects (instances) of the class
student1 = Student("Ali", 20, "A")
student2 = Student("Sara", 16, "B")

# Calling methods on the objects
student1.display_info()
print("Is adult:", student1.is_adult())

print()

student2.display_info()
print("Is adult:", student2.is_adult())


"""
splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=250,
    chunk_overlap=0
)

res=splitter.split_text(text)
print(res)