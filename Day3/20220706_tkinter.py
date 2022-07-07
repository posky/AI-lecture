from tkinter import *
import pandas as pd
import os

global dat

print(os.getcwd())
path = r'D:\dev\AI-lecture\Day3'
os.chdir(path)
print(os.getcwd())

dat = pd.read_excel('dic_excel.xlsx')
print(dat.shape, dat)
print()

def click():
    word = entry.get()
    output.delete(0.0, END)     # 내용 지우기 (text)

    try:
        def_word = dat.loc[dat['word'] == word, 'def'].values[0]
    except:
        def_word = '단어의 뜻을 찾을 수 없습니다.'

    output.insert(END, def_word)


dat2 = pd.read_csv('dic_csv.csv')
print(dat2.shape, dat2)

window = Tk()
window.title('My Dictionary')

label = Label(window, text='원하는 단어 입력 후, 엔터 키 누르기')
label.grid(row = 0, column = 0, sticky = W)

entry = Entry(window, width = 15, bg = 'light green')
entry.grid(row = 1, column = 0, sticky = W)

btn = Button(window, text='검색', width = 5, command = click)
btn.grid(row = 2, column = 0, sticky = W)

label2 = Label(window, text = '\n의미')
label2.grid(row = 3, column = 0, sticky = W)

output = Text(window, width=50, height=6, wrap=WORD, background='light green')
output.grid(row=4, column=0, columnspan=2, sticky=W)


window.mainloop()