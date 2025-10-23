import tkinter

class App():
    def __init__(self):
        self.rootTK = tkinter.Tk()
        self.rootTK.title = "Emotion ERC"
        self.rootTK.minsize(1000, 400)
        self.row = 0
        self.chatFrame = self.createChatBox()
        self.TextInput = self.createChatInput()
        self.createText("debug test whatever i am the bot or whatever", False)
        self.rootTK.mainloop()
    def createChatBox(self):
        chatFrame = tkinter.Frame(self.rootTK, bg="lightyellow", relief="solid", borderwidth=1)
        chatFrame.place(x=0, y=0, relwidth=1, relheight=.7)
        chatFrame.grid_columnconfigure(0, weight=1)
        chatFrame.grid_columnconfigure(1, weight=1)
        return chatFrame
    def createChatInput(self):
        chatInputFrame = tkinter.Frame(self.rootTK, borderwidth=3)

        chatInputFrame.grid_columnconfigure(0, weight=1)
        chatInputFrame.grid_columnconfigure(1, weight=10)
        chatInputFrame.grid_columnconfigure(5, weight=1)
        chatInputFrame.grid_rowconfigure(0, weight=1)

        chatInputFrame.place(x=0, rely=.7, relwidth=1, relheight=.3)

        chatLabel = tkinter.Label(chatInputFrame, text="Chat:")
        Text = tkinter.Text(chatInputFrame)
        TextButton = tkinter.Button(chatInputFrame, text="Chat!", command=lambda: self.createText(Text.get("1.0", "end-1c") , True))

        chatLabel.grid(column=0, row=0, columnspan=1)
        Text.grid(column=1, row=0, columnspan=4)
        TextButton.grid(column=2, row=0)
        Text.focus_set()
        return Text
    def createText(self, txt, isUser):
        Text = tkinter.Text(self.chatFrame, borderwidth=3)
        if isUser:
            Text.grid(column=1, row=self.row, sticky="ew")
        else:
            Text.grid(column=0, row=self.row, sticky="ew")
        Text.insert(tkinter.END, txt)
        line_count = int(Text.tk.call(Text, 'count', '-lines', '1.0', 'end - 1c'))
    
        Text.config(height=line_count if line_count > 0 else 1)
        Text.config(state=tkinter.DISABLED)
        self.row += 1
if __name__ == '__main__':
    app = App()
