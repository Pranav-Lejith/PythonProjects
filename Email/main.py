import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('pranavlejith@gmail.com','Monday@124')
server.sendmail('pranavlejith@gmail.com',
                'pranavlejith124@gmail.com'
                'Auto message from pytjpon')