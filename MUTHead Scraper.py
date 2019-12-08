
from time import sleep
from multiprocessing import Process
import threading

from bs4 import BeautifulSoup
import pandas as pd
import requests

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Getting Basic info and link to Player Page
data = []
home_url = "https://www.muthead.com"
features = []

overview_stats = ["PLAYER_NAME", "OVR", "POS", "PROGRAM"]
card_stats = ["TEAM", "HEIGHT", "WEIGHT", "ARCHETYPE", "PRICE", "QUICKSELL", "QS_CURRENCY"]
player_traits = ["clutch", "penalty", "lb_style", "dl_swim", "dl_spin", "dl_bull", "big_hitter", "strips_ball",
                 "ball_in_air", "high_motor", "covers_ball", "extra_yards", "agg_catch", "rac_catch",
                 "poss_catch", "drops_open", "sideline_catch", "qb_style", "tight_spiral", "sense_pressure",
                 "throw_away", "force_passes"]
gen_attr = ["SPD", "STR", "AGI", "ACC", "AWA", "CTH", "JMP", "STA", "INJ"]
off_attr = ["TRK", "ELU", "BTK", "BCV", "SFA", "SPM", "JKM", "CAR", "SRR", "MRR", "DRR", "CIT", "SPC", "RLS",
            "THP", "SAC", "MAC", "DAC", "RUN", "TUP", "BSK", "PAC", "RBK", "RBP", "RBF", "PBK", "PBP", "PBF",
            "LBK", "IBL"]
def_attr = ["TAK", "POW", "PMV", "FMV", "BSH", "PUR", "PRC", "MCV", "ZCV", "PRS"]
st_attr = ["KPW", "KAC", "RET"]
features = overview_stats + card_stats + player_traits + gen_attr + off_attr + def_attr + st_attr

#  Setting Selenium Browser Stuff
prefs = {"profile.managed_default_content_settings.images": 2}
chrome_options = Options()
#chrome_options.add_argument("--headless")
chrome_options.add_extension('UblockOrigin.crx')
#driver = webdriver.Chrome('/usr/lib/chromium-browser/chromedriver', chrome_options=chrome_options)

chrome_options.add_experimental_option("prefs", prefs)

browser1 = webdriver.Chrome(options=chrome_options)
browser2 = webdriver.Chrome(options=chrome_options)
browser3 = webdriver.Chrome(options=chrome_options)


def grab_data(browser_num, start_page):
    for page_number in range(start_page, 4, 3):
        # Get all the pages of all the players
        page_url = home_url + "/20/players?page=" + str(page_number)
        r = requests.get(page_url)  # returns a response
        content = r.text
        soup = BeautifulSoup(content, features="html.parser")
        players = soup.findAll("li", class_="player-listing__item")

        for player in players:
            player_data = []
            # OVERVIEW STATS
            ref_page_num = player.find("a").get("href")
            OVR = int(player.span.contents[0].strip())
            PLAYER_NAME = str(player.find("div", class_="list-info-player__player-name").contents[0].strip())
            POS = str(player.find("div", class_="list-info-player__player-info").contents[0].split("|")[0].strip())
            PROGRAM = str(player.find("div", class_="list-info-player__player-info").contents[0].split("|")[1].strip())

            # Get Card Page
            player_url = home_url + ref_page_num
            player_r = requests.get(player_url)
            player_content = player_r.text
            player_soup = BeautifulSoup(player_content, features="html.parser")

            # CARD STATS
            TEAM = player_soup.find("a", class_="mut-player-summary__team").contents[0].strip()
            HEIGHT = str(player_soup.find("span", class_="mut-player-summary__height").contents[0][4:])
            HEIGHT_INCHES = int(HEIGHT[0])*12 + int(HEIGHT[3])
            WEIGHT = player_soup.find("span", class_="mut-player-summary__weight").contents[0][3:]
            ARCHETYPE = str(player_soup.find(text="Archetype").parent.parent.parent.p.contents[0])
            PRICE = player_soup.find(class_="mut-player-price__price").contents[0]
            if 'K' in PRICE:
                PRICE = int(float(PRICE.replace('K', ''))* 1000)
            elif 'M' in PRICE:
                 PRICE = int(float(PRICE.replace('M', '')) * 1000000)
            elif '—' in PRICE:
                PRICE = 'Not Auctionable'
            QUICKSELL = str(player_soup.find(class_="infobox__statline-left").contents[0])
            QS_CURRENCY = str(player_soup.find(class_="infobox__statline-right").contents[0])

            # Assign Player Traits [NEED TO MAKE MORE READABLE using a dictionary, or a tuple]
            try:
                clutch = str(player_soup.find(class_="infobox__statline-left",
                                    text="Clutch").find_next_sibling().contents[0])
            except AttributeError:
                clutch = "N/A"

            try:
                penalty = str(player_soup.find(class_="infobox__statline-left",
                                text="Penalty").find_next_sibling().contents[0])
            except AttributeError:
                penalty = "N/A"

            try:
                lb_style = str(player_soup.find(class_="infobox__statline-left",
                                text="LB Style").find_next_sibling().contents[0])
            except AttributeError:
                lb_style = "N/A"

            try:
                dl_swim = str(player_soup.find(class_="infobox__statline-left",
                                text="DL Swim").find_next_sibling().contents[0])
            except AttributeError:
                dl_swim = "N/A"

            try:
                dl_spin = str(player_soup.find(class_="infobox__statline-left",
                                text="DL Spin").find_next_sibling().contents[0])
            except AttributeError:
                dl_spin = "N/A"

            try:
                dl_bull = str(player_soup.find(class_="infobox__statline-left",
                                text="DL Bull").find_next_sibling().contents[0])
            except AttributeError:
                dl_bull = "N/A"

            try:
                big_hitter = str(player_soup.find(class_="infobox__statline-left",
                                text="Big Hitter").find_next_sibling().contents[0])
            except AttributeError:
                big_hitter = "N/A"

            try:
                strips_ball = str(player_soup.find(class_="infobox__statline-left",
                                text="Strips Ball").find_next_sibling().contents[0])
            except AttributeError:
                strips_ball = "N/A"

            try:
                ball_in_air = str(player_soup.find(class_="infobox__statline-left",
                                text="Plays Ball in Air").find_next_sibling().contents[0])
            except AttributeError:
                ball_in_air = "N/A"

            try:
                high_motor = str(player_soup.find(class_="infobox__statline-left",
                                            text="High Motor").find_next_sibling().contents[0])
            except AttributeError:
                high_motor = "N/A"

            try:
                covers_ball = str(player_soup.find(class_="infobox__statline-left",
                                text="Covers Ball").find_next_sibling().contents[0])
            except AttributeError:
                covers_ball = "N/A"

            try:
                extra_yards = str(player_soup.find(class_="infobox__statline-left",
                                text="First for Extra Yards").find_next_sibling().contents[0])
            except AttributeError:
                extra_yards = "N/A"

            try:
                agg_catch = str(player_soup.find(class_="infobox__statline-left",
                                text="Makes Aggressive Catches").find_next_sibling().contents[0])
            except AttributeError:
                agg_catch = "N/A"

            try:
                rac_catch = str(player_soup.find(class_="infobox__statline-left",
                                text="Makes RAC Catches").find_next_sibling().contents[0])
            except AttributeError:
                rac_catch = "N/A"

            try:
                poss_catch = str(player_soup.find(class_="infobox__statline-left",
                                text="Makes Possession Catches").find_next_sibling().contents[0])
            except AttributeError:
                poss_catch = "N/A"

            try:
                drops_open = str(player_soup.find(class_="infobox__statline-left",
                                text="Drops Open Passes").find_next_sibling().contents[0])
            except AttributeError:
                drops_open = "N/A"

            try:
                sideline_catch = str(player_soup.find(class_="infobox__statline-left",
                                text="Makes Sideline Catches").find_next_sibling().contents[0])
            except AttributeError:
                sideline_catch = "N/A"

            try:
                qb_style = str(player_soup.find(class_="infobox__statline-left",
                                text="QB Style").find_next_sibling().contents[0])
            except AttributeError:
                qb_style = "N/A"

            try:
                tight_spiral = str(player_soup.find(class_="infobox__statline-left",
                                text="Throws Tight Spiral").find_next_sibling().contents[0])
            except AttributeError:
                tight_spiral = "N/A"

            try:
                sense_pressure = str(player_soup.find(class_="infobox__statline-left",
                                text="Senses Pressure").find_next_sibling().contents[0])
            except AttributeError:
                sense_pressure = "N/A"

            try:
                throw_away = str(player_soup.find(class_="infobox__statline-left",
                                text="Throws Ball Away").find_next_sibling().contents[0])
            except AttributeError:
                throw_away = "N/A"

            try:
                force_passes = str(player_soup.find(class_="infobox__statline-left",
                                text="Forces Passes").find_next_sibling().contents[0])
            except AttributeError:
                force_passes = "N/A"

            eval(browser_num).get(player_url.replace("players", "compare"))
            # GEN ATTRIBUTES
            SPD = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[2]/table/tbody/tr[1]/td[1]').text
            STR = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[2]/table/tbody/tr[2]/td[1]').text
            AGI = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[2]/table/tbody/tr[3]/td[1]').text
            ACC = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[2]/table/tbody/tr[4]/td[1]').text
            AWA = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[2]/table/tbody/tr[5]/td[1]').text
            CTH = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[2]/table/tbody/tr[6]/td[1]').text
            JMP = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[2]/table/tbody/tr[7]/td[1]').text
            STA = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[2]/table/tbody/tr[8]/td[1]').text
            INJ = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[2]/table/tbody/tr[9]/td[1]').text
            # OFF ATTRIBUTES
            TRK = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[1]/td[1]').text
            ELU = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[2]/td[1]').text
            BTK = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[3]/td[1]').text
            BCV = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[4]/td[1]').text
            SFA = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[5]/td[1]').text
            SPM = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[6]/td[1]').text
            JKM = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[7]/td[1]').text
            CAR = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[8]/td[1]').text
            SRR = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[9]/td[1]').text
            MRR = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[10]/td[1]').text
            DRR = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[11]/td[1]').text
            CIT = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[12]/td[1]').text
            SPC = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[13]/td[1]').text
            RLS = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[14]/td[1]').text
            THP = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[15]/td[1]').text
            SAC = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[16]/td[1]').text
            MAC = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[17]/td[1]').text
            DAC = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[18]/td[1]').text
            RUN = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[19]/td[1]').text
            TUP = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[20]/td[1]').text
            BSK = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[21]/td[1]').text
            PAC = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[22]/td[1]').text
            RBK = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[23]/td[1]').text
            RBP = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[24]/td[1]').text
            RBF = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[25]/td[1]').text
            PBK = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[26]/td[1]').text
            PBP = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[27]/td[1]').text
            PBF = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[28]/td[1]').text
            LBK = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[29]/td[1]').text
            IBL = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[3]/table/tbody/tr[30]/td[1]').text

            # DEF ATTRIBUTES
            TAK = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[4]/table/tbody/tr[1]/td[1]').text
            POW = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[4]/table/tbody/tr[2]/td[1]').text
            PMV = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[4]/table/tbody/tr[3]/td[1]').text
            FMV = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[4]/table/tbody/tr[4]/td[1]').text
            BSH = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[4]/table/tbody/tr[5]/td[1]').text
            PUR = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[4]/table/tbody/tr[6]/td[1]').text
            PRC = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[4]/table/tbody/tr[7]/td[1]').text
            MCV = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[4]/table/tbody/tr[8]/td[1]').text
            ZCV = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[4]/table/tbody/tr[9]/td[1]').text
            PRS = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[4]/table/tbody/tr[10]/td[1]').text

            # SPECIAL TEAM ATTRIBUTES
            KPW = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[5]/table/tbody/tr[1]/td[1]').text
            KAC = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[5]/table/tbody/tr[2]/td[1]').text
            RET = eval(browser_num).find_element_by_xpath(
                '//*[@id="slideout__panel"]/div[2]/div[2]/div/div[5]/table/tbody/tr[3]/td[1]').text

            for feature in features:
                player_data.append(eval(feature))
            print(player_data)
            data.append(player_data)
    eval(browser_num).quit()


if __name__=='__main__':
    thread1 = threading.Thread(target=grab_data, args=("browser1", 1))
    thread2 = threading.Thread(target=grab_data, args=("browser2", 2))
    thread3 = threading.Thread(target=grab_data, args=("browser3", 3))
    thread1.start()
    thread2.start()
    thread3.start()
    thread1.join()
    thread2.join()
    thread3.join()
    print("All other threads are finished")
df = pd.DataFrame(data, columns=features)
df.to_csv("MutData.csv")
