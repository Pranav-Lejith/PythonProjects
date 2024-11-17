import pywhatkit
client = str(input("Enter the phone number of the receiver(With the country code): "))
msg = str(input("Enter the message to be sent: "))
hour = int(input("Enter the hour on which the message is to be sent(In 24 hour clock): "))
minute = int(input("Enter the minute of the hour on which the message is to be sent: "))
pywhatkit.sendwhatmsg(client,msg,hour,minute)       