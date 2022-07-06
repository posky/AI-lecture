from tkinter import *
import pandas as pd
import os

def click():
    pass


global dat

print(os.getcwd())
path = r'D:\dev\AI-lecture\Day3'
os.chdir(path)
print(os.getcwd())

dat = pd.read_excel('dic_excel.xlsx')
print(dat.shape, dat)
print()


dat2 = pd.read_csv('dic_csv.csv')
print(dat2.shape, dat2)

window = Tk()
window.title('My Dictionary')

label = Label(window, text='원하는 단어 입력 후, 엔터 키 누르기')
label.grid(row = 0, column = 0, sticky = W)

entry = Entry(window, width = 15, bg = 'light green')
entry.grid(row = 1, column = 0, sticky = W)

btn = Button(window, text='제출', width = 5, command = click)
btn.grid(row = 2, column = 0, sticky = W)

label2 = Label(window, text = '\n의미')
label2.grid(row = 3, column = 0, sticky = W)


window.mainloop()