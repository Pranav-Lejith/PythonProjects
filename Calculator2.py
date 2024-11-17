def add(n1,n2):
    print("Answer = ",n1+n2)

def subtract(n1,n2):
    if n1 > n2:
        print("Answer = ",n1-n2)
    elif n2 < n1:
        print("Answer = ",n2-n1)

def multiply(n1,n2):
    print("Answer = ",n1*n2)

def divide(n1,n2):
    if n1 > n2:
        print("Answer = ",n1/n2)
    elif n2 < n1:
        print("Answer = ",n2/n1)

x = int(input("Enter first number: "))
y = int(input("Enter Second Number: "))
choice = str(input("Enter operation: "))

if choice == '+':
    add(x,y)   

elif choice == '-':
    subtract(x,y)

elif choice == '*':
    multiply(x,y)

elif choice == '/':
    divide(x,y)            

else:
    print("You have given a wrong choice.")    


    