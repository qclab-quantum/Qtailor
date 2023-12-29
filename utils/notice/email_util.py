
import smtplib
class EmailUtil():

    def __init__(self, ):
        # SMTP服务器的登录信息
        smtp_server = 'smtp.163.com'
        smtp_port = 25
        smtp_username = 'atiandev@163.com'
        smtp_password = 'your_password'

        # 发件人和收件人的邮箱地址
        from_email = 'atiandev@163.com'

        def send(to_email='904715458@qq.com', subject='', message=''):
            email_content = f'Subject: {subject}\n\n{message}'
            # 与SMTP服务器建立连接
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                # 发送邮件
                server.sendmail(from_email, to_email, email_content)

            print('Email sent successfully.')

if __name__ == '__main__':
    EmailUtil().send('')