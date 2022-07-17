from cgitb import small
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import WebDriverException
from time import sleep
import random

# Khai báo biến browser
browser  = webdriver.Chrome(executable_path="./chromedriver")

# Mở trang web facebook
browser.get("http://facebook.com")

# Điền thông tin vào ô user và pass
txtUser = browser.find_element_by_id("email")
txtUser.send_keys("chanjohnwick@gmail.com") # <---  Điền username thật của các bạn vào đây
txtPass = browser.find_element_by_id("pass")
txtPass.send_keys("chimcodon")

# Submit form
txtPass.send_keys(Keys.ENTER)
sleep(random.randint(2,3))

linkPost = [
    "https://www.facebook.com/Theanh28/photos/a.1509442759101639/6063276670384869/",
    "https://www.facebook.com/photo/?fbid=559130745579806&set=a.287103319449218",
    "https://www.facebook.com/hhsb.vn/photos/a.1883296621685362/5996910276990622/",
    "https://www.facebook.com/SadnessMusicVideo/photos/a.372226893611795/1279792132855262",
    "https://www.facebook.com/kinhte.kienthuc/photos/a.620263914708901/5094846610583920/",
    "https://www.facebook.com/ThichXemBongDa.Page/photos/a.443641709734325/1205531630211992/",
    "https://www.facebook.com/hhsb.vn/photos/a.1883296621685362/5992338257447824/",
    "https://www.facebook.com/24hstore.vn/photos/a.1022921527742168/5472131862821090/",
    "https://www.facebook.com/neuconfessions/photos/a.896001240449769/5631223136927532",
    "https://www.facebook.com/neuconfessions/photos/a.896001240449769/5627715397278306/",
    "https://www.facebook.com/neuconfessions/photos/a.896001240449769/5624785664237946/",
    "https://www.facebook.com/neuconfessions/photos/a.896001240449769/5617758944940618/",
    "https://www.facebook.com/cambongda/photos/a.274738456750617/1079275142963607/",
    "https://www.facebook.com/cambongda/photos/a.274738456750617/1077948033096318/",
]

for i in linkPost:
    # Mở URL của post
    browser.get(i)
    sleep(random.randint(4,5))

    # dropdown loại comment
    dropdown_comment_type = browser.find_element_by_xpath("/html/body/div[1]/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div[2]/div/div/div/div[1]/div[4]/div[1]/div/div/div/span/div/div/i")
    dropdown_comment_type.click()
    sleep(random.randint(1,3))

    # chọn all comment
    choose_all_comment = browser.find_element_by_xpath("/html/body/div[1]/div/div[1]/div/div[3]/div/div/div[2]/div/div/div[1]/div[1]/div/div/div[1]/div/div/div/div[1]/div/div[3]/div[1]")
    choose_all_comment.click()
    sleep(random.randint(2,4))


    # show more comment
    count = 0
    print(i)
    for j in range(1,10,1):
        try:
            show_more_comment = browser.find_element_by_xpath("/html/body/div[1]/div/div[1]/div/div[3]/div/div/div[1]/div[1]/div/div[2]/div/div/div/div[1]/div[4]/div[2]/div[1]/div[2]")
            show_more_comment.click()
            sleep(random.randint(4,6))
        except WebDriverException:
            count += 1
            sleep(random.randint(2,3))
            continue
    sleep(random.randint(5,10))

    # Tìm tất cả các comment và ghi ra màn hình (hoặc file)
    # -> lấy all thẻ div có thuộc tính role="article"
    comment_list = browser.find_elements(By.XPATH, '//div[@role="article"]')

    # Lặp trong tất cả các comment và hiển thị nội dung comment ra màn hình
    clone = []
    for comment in comment_list:
        poster = comment.find_element_by_class_name("d2edcug0")
        clone.append(poster.text)

    text2 = []
    for k in clone:
        k = k.split('\n  ·\nTheo dõi\n')
        if len(k) < 2:
            k = k[0].split('\n')
        text2.append(k[-1])


    # with open('comments.txt', "w", encoding="utf-8") as file:
    with open('cmt.txt', "a", encoding="utf-8") as file:
        for line in text2:
            file.writelines(line+"\n")

    print(count)

sleep(random.randint(4,7))
# Đóng browser
browser.close()