import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class EmailNotifier:
    def __init__(self, smtp_server, port, sender_email, password):
        """
        初始化邮件通知器
        :param smtp_server: SMTP服务器地址
        :param port: SMTP服务器端口
        :param sender_email: 发送者邮箱
        :param password: 发送者邮箱密码
        """
        self.smtp_server = smtp_server
        self.port = port
        self.sender_email = sender_email
        self.password = password

    def send_email(self, recipient_email, subject, body):
        """
        发送邮件
        :param recipient_email: 接收者邮箱
        :param subject: 邮件主题
        :param body: 邮件正文
        """
        # 创建邮件对象
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # 添加邮件正文
        msg.attach(MIMEText(body, 'plain'))

        # 连接到SMTP服务器
        with smtplib.SMTP_SSL(self.smtp_server, self.port) as server:
            server.login(self.sender_email, self.password)
            text = msg.as_string()
            server.sendmail(self.sender_email, recipient_email, text)

def demo():
    SMTP_SERVER = "smtp.163.com"
    PORT = 465  # 对于SSL连接
    SENDER_EMAIL = "notify_helper@163.com"
    PASSWORD = "TRBWNPDOPJDYUFSZ"
    # 创建EmailNotifier实例
    notifier = EmailNotifier(SMTP_SERVER, PORT, SENDER_EMAIL, PASSWORD)
    # 发送邮件
    notifier.send_email("youremail@qq.com", "实验完成通知", "这是邮件正文。")
class Notifier:
    def __init__(self):
        SMTP_SERVER = "smtp.163.com"
        PORT = 465  # 对于SSL连接
        SENDER_EMAIL = "notify_helper@163.com"
        PASSWORD = "TRBWNPDOPJDYUFSZ"
        self.notifier = EmailNotifier(SMTP_SERVER, PORT, SENDER_EMAIL, PASSWORD)

    def on_experiment_finsh(self,email,subject="实验完成通知", body="实验完成"):
        self.notifier.send_email(email,subject,body)

# 使用示例
if __name__ == "__main__":

    Notifier().on_experiment_finsh('email@email.com',subject="实验完成" +'amplitude_estimation/ae_indep_qiskit_20.qasm', body="实验完成" + 'amplitude_estimation/ae_indep_qiskit_20.qasm')