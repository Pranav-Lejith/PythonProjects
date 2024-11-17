import speech_recognition as sr
import pyttsx3
import tkinter as tk
import threading
import os

class VirtualAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.window = None

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
            return audio

    def recognize_command(self):
        try:
            audio = self.listen()
            command = self.recognizer.recognize_google(audio)
            print(f"You said: {command}")
            if "hey bot" in command.lower():
                self.open_window()
            # Handle other commands here
        except Exception as e:
            print(f"Error: {e}")

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def open_window(self):
        if self.window is None:
            self.window = tk.Tk()
            self.window.title("Virtual Assistant")
            self.window.geometry("300x200")

            label = tk.Label(self.window, text="Hello! I'm your assistant.", font=("Arial", 16))
            label.pack(pady=20)

            btn = tk.Button(self.window, text="Close", command=self.close_window)
            btn.pack(pady=10)

            self.window.protocol("WM_DELETE_WINDOW", self.close_window)
            self.window.mainloop()

    def close_window(self):
        if self.window is not None:
            self.window.destroy()
            self.window = None

    def run(self):
        while True:
            self.recognize_command()

if __name__ == "__main__":
    assistant = VirtualAssistant()
    threading.Thread(target=assistant.run, daemon=True).start()
