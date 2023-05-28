import snscrape.modules.twitter as sntwitter
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
import multiprocessing as mult

maxtweets_per_trial = 5000

def batch1():
    ini_date = date(2022,3,15)
    for game in range(269, 272):
        tweets_list2 = []
        for trials in range(1,7):
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper('"Wordle '+str(game)+' '+str(trials)+'/6"'
                                                                    +' lang:en since:'+str(ini_date)
                                                                    +' until:'+ str(ini_date + timedelta(days=1))).get_items()):
                if i>maxtweets_per_trial:
                    break
                tweets_list2.append([game , trials ,tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.url])
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper('"Wordle '+str(game)+' X/6"'
                                                                    +' lang:en since:'+str(ini_date)
                                                                    +' until:'+ str(ini_date + timedelta(days=1))).get_items()):
                if i>maxtweets_per_trial:
                    break
                tweets_list2.append([game , 'X' ,tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.url])
        ini_date = ini_date + timedelta(days=1)
        tweets_df2 = pd.DataFrame(tweets_list2, columns=['Game', 'Trial' ,'Datetime', 'Tweet Id', 'Text', 'Username', 'URL'])
        tweets_df2.to_csv('../scraping_raw/' + str(game) + '.csv')

def batch2():
    ini_date = date(2022,3,18)
    for game in range(272, 275):
        tweets_list2 = []
        for trials in range(1,7):
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper('"Wordle '+str(game)+' '+str(trials)+'/6"'
                                                                    +' lang:en since:'+str(ini_date)
                                                                    +' until:'+ str(ini_date + timedelta(days=1))).get_items()):
                if i>maxtweets_per_trial:
                    break
                tweets_list2.append([game , trials ,tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.url])
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper('"Wordle '+str(game)+' X/6"'
                                                                    +' lang:en since:'+str(ini_date)
                                                                    +' until:'+ str(ini_date + timedelta(days=1))).get_items()):
                if i>maxtweets_per_trial:
                    break
                tweets_list2.append([game , 'X' ,tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.url])
        ini_date = ini_date + timedelta(days=1)
        tweets_df2 = pd.DataFrame(tweets_list2, columns=['Game', 'Trial' ,'Datetime', 'Tweet Id', 'Text', 'Username', 'URL'])
        tweets_df2.to_csv('../scraping_raw/' + str(game) + '.csv')

def batch3():
    ini_date = date(2022,3,21)
    for game in range(275, 277):
        tweets_list2 = []
        for trials in range(1,7):
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper('"Wordle '+str(game)+' '+str(trials)+'/6"'
                                                                    +' lang:en since:'+str(ini_date)
                                                                    +' until:'+ str(ini_date + timedelta(days=1))).get_items()):
                if i>maxtweets_per_trial:
                    break
                tweets_list2.append([game , trials ,tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.url])
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper('"Wordle '+str(game)+' X/6"'
                                                                    +' lang:en since:'+str(ini_date)
                                                                    +' until:'+ str(ini_date + timedelta(days=1))).get_items()):
                if i>maxtweets_per_trial:
                    break
                tweets_list2.append([game , 'X' ,tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.url])
        ini_date = ini_date + timedelta(days=1)
        tweets_df2 = pd.DataFrame(tweets_list2, columns=['Game', 'Trial' ,'Datetime', 'Tweet Id', 'Text', 'Username', 'URL'])
        tweets_df2.to_csv('../scraping_raw/' + str(game) + '.csv')
        
def batch4():
    ini_date = date(2022,10,7)
    for game in range(475, 516):
        tweets_list2 = []
        for trials in range(1,7):
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper('"Wordle '+str(game)+' '+str(trials)+'/6"'
                                                                    +' lang:en since:'+str(ini_date)
                                                                    +' until:'+ str(ini_date + timedelta(days=1))).get_items()):
                if i>maxtweets_per_trial:
                    break
                tweets_list2.append([game , trials ,tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.url])
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper('"Wordle '+str(game)+' X/6"'
                                                                    +' lang:en since:'+str(ini_date)
                                                                    +' until:'+ str(ini_date + timedelta(days=1))).get_items()):
                if i>maxtweets_per_trial:
                    break
                tweets_list2.append([game , 'X' ,tweet.date, tweet.id, tweet.content, tweet.user.username, tweet.url])
        ini_date = ini_date + timedelta(days=1)
        tweets_df2 = pd.DataFrame(tweets_list2, columns=['Game', 'Trial' ,'Datetime', 'Tweet Id', 'Text', 'Username', 'URL'])
        tweets_df2.to_csv('../scraping_raw/' + str(game) + '.csv')