import pickle
import sys

class Book:
    def __init__(self, books_dictionary):
        self.books_dictionary = books_dictionary

    def store_data(self):
        self.books_data = pickle.dumps(self.books_dictionary)
        return self.books_data


class Student:
    def __init__(self, students_dictionary):
        self.students_dictionary = students_dictionary

    def store_data(self):
        students_data = pickle.dumps(self.students_dictionary)
        return students_data

    def requestBook(self):
        print("Enter the name of the book you'd like to borrow: ")
        self.book=input()
        return self.book

    def returnBook(self):
        print("Enter the name of the book you'd like to return: ")
        self.book=input()
        return self.book


class Library:
    def __init__(self):
        self.all_books = {}

    def display_available_books(self, book):
        self.book = book
        self.all_books = pickle.loads(self.book.store_data())
        print("The books in the library are as follows:-")
        for value in self.all_books.values():
            print(value)
        return self.all_books

    def lend_book(self, requested_book):
        self.requested_book = requested_book
        if self.requested_book in self.all_books.values():
            print("The book you requested has now been borrowed")

            for key, value in self.all_books.items():
                if requested_book == value:
                    x = key
            self.all_books.pop(x)
            print(self.all_books)

        else:
            print("Sorry the book you requested is not in the library")
                  
    def receive_book(self, returned_book, book):
        self.book = book
        self.orignal_books = pickle.loads(self.book.store_data())

        i = 1
        for value in self.orignal_books.values():
            if (returned_book == value):
                break
            else:
                i +=1

        self.all_books.update({i : returned_book})
        print(self.all_books)
            

def main():
    book = Book({1 : "Harry Potter", 2 : "Manga", 3 : "The Invisible Man"})
    student = Student({1 : "Ronit", 2 : "Rohit", 3 : "Rahul", 4 : "Rohan"})
    library = Library()
    done=False
    while done==False:
        print("""
            1. Display all available books
            2. Request a book
            3. Return a book
            4. Exit
            """)
        choice = int(input("Enter Choice:"))
        
        if choice == 1:
            library.display_available_books(book)
        elif choice == 2:
            library.lend_book(student.requestBook())
        elif choice == 3:
            library.receive_book(student.returnBook(), book)
        elif choice == 4:
            sys.exit()
                  
main()
