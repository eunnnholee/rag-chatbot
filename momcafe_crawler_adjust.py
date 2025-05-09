import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import chromedriver_autoinstaller


class MomcafeCrawler:
    def __init__(self):
        chromedriver_autoinstaller.install()
        opts = Options()
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(service=Service(), options=opts)
        self.wait = WebDriverWait(self.driver, 10)
        self.driver.implicitly_wait(5)

    def manual_login(self):
        print("[LOGIN] 네이버 로그인 페이지로 이동합니다.")
        self.driver.get("https://nid.naver.com/nidlogin.login")
        input("로그인 완료 후, 로그인 완료하면 이 창에서 Enter를 눌러주세요...")

    def open_board(self, board_xpath: str):
        self.driver.get("https://cafe.naver.com/imsanbu")
        time.sleep(2)

        menu_el = self.wait.until(
            EC.presence_of_element_located((By.XPATH, board_xpath)))
        board_href = menu_el.get_attribute("href")

        board_url = "https://cafe.naver.com" + \
            board_href if board_href.startswith("/") else board_href
        self.driver.get(board_url)
        time.sleep(2)
        print('[MOVE] 게시판 이동 완료')

        self.driver.switch_to.default_content()
        time.sleep(1)
        try:
            self.driver.switch_to.frame("cafe_main")
            print("[FRAME] cafe_main으로 전환 완료")
        except:
            print("[FRAME] cafe_main 없음, 그냥 진행")

        self.board_url = board_url

    def crawl_board(self, board_name: str, board_xpath: str, post_xpath: str, pages: int, keyword: str = "육아휴직", start_page: int = 1):
        self.open_board(board_xpath)

        titles = []
        details = []

        try:
            engine = self.driver.find_element(
                By.XPATH, '//*[@id="cafe_content"]/div[5]/div[2]/div[3]/input')
            engine.click()
            engine.send_keys(keyword)
            engine.send_keys(Keys.ENTER)
            print('[ENTER] 검색어 입력 완료')
            time.sleep(2)
        except Exception as e:
            print(f"[ERROR] 검색창 입력 실패: {e}")
            return []

        for page in range(1, pages + 1):
            print(f"[{board_name}] {page} 페이지 크롤링 중...")

            if page < start_page:
                print(f"[SKIP] {page} 페이지는 건너뜁니다.")
            else:
                for i in range(1, 16):
                    try:
                        post = self.driver.find_element(
                            By.XPATH, f'//*[@id="cafe_content"]/div[4]/table/tbody/tr[{i}]/td[2]/div/div/a[1]').get_attribute('href')
                        self.driver.get(post)
                        time.sleep(2)

                        self.driver.switch_to.default_content()
                        try:
                            self.driver.switch_to.frame("cafe_main")
                        except:
                            pass

                        title = self.driver.find_element(
                            By.XPATH, '//*[@id="app"]/div/div/div[2]/div[1]/div[1]/div/div/h3').text
                        print('제목: ' + title)
                        titles.append(title)

                        try:
                            detail = self.driver.find_element(
                                By.XPATH, '//*[@id="app"]/div/div/div[2]/div[2]/div[1]/div/div[2]').text
                        except:
                            detail = ""
                            print("내용 없음")

                        print('내용: ' + detail)
                        details.append(detail)

                        time.sleep(1)
                        self.driver.back()
                        print('finish one post')

                    except Exception as e:
                        print(f"오류 발생: {e}")
                        self.driver.back()
                        time.sleep(1)

            # 페이지 넘기기 로직 (그대로 유지)
            if page < pages:
                if page <= 10:
                    if page % 10 != 0:
                        self.driver.find_element(
                            By.XPATH, f'//*[@id="cafe_content"]/div[6]/div[1]/button[{page % 10 + 1}]').click()
                    else:
                        self.driver.find_element(
                            By.XPATH, f'//*[@id="cafe_content"]/div[6]/div[1]/button[11]').click()
                else:
                    if page % 10 != 0:
                        self.driver.find_element(
                            By.XPATH, f'//*[@id="cafe_content"]/div[6]/div[1]/button[{page % 10 + 2}]').click()
                    else:
                        self.driver.find_element(
                            By.XPATH, f'//*[@id="cafe_content"]/div[6]/div[1]/button[12]').click()

        return titles, details

    def close(self):
        self.driver.quit()


# 메인 실행
if __name__ == "__main__":

    all_titles = []
    all_details = []

    crawler = MomcafeCrawler()
    crawler.manual_login()

    boards = [
        ("임신중 질문방", '//*[@id="menuLink392"]',
         '//*[@id="main-area"]/div[5]/table/tbody/tr[*]/td[1]/div[2]/div/a', 60),
        ("육아 질문방",   '//*[@id="menuLink126"]',
         '//*[@id="main-area"]/div[5]/table/tbody/tr[*]/td[1]/div[2]/div/a', 20),
        ("자유 수다방",   '//*[@id="menuLink179"]',
         '//*[@id="main-area"]/div[5]/table/tbody/tr[*]/td[1]/div[2]/div/a', 90),
    ]

    try:
        for name, board_xpath, post_xpath, page_cnt in boards:
            if name == "임신중 질문방":
                # 1~19페이지는 skip, 20~60 크롤링
                titles, details = crawler.crawl_board(
                    name, board_xpath, post_xpath, pages=60, start_page=21, keyword="육아휴직")
            else:
                titles, details = crawler.crawl_board(
                    name, board_xpath, post_xpath, pages=page_cnt, keyword="육아휴직")

            all_titles.extend(titles)
            all_details.extend(details)

    except Exception as e:
        print(f"[ERROR] 전체 크롤링 도중 예외 발생: {e}")
    finally:
        df = pd.DataFrame({'title': all_titles, 'detail': all_details})
        df.to_csv("navercafe_results_partial2.csv",
                  index=False, encoding="utf-8-sig")
        print(f"[SAVE] 크롤링 중단 시점까지 저장 완료: 총 {len(df)}건")
        crawler.close()
