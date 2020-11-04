# Madden Ultimate Team

## What is Madden Ultimate Team? :video_game:
* Madden Ultimate Team is a popular mode in EA's Madden NFL video game series. 
* EA releases new versions of every card for every NFL athlete as the year progresses with the card stats generally improving until the end of the game's life cycle. 
* Cards can be bought for in-game currency on the auction house, or real money, and then used in a game of simulation football to play against other players across the world. 
*With hundreds of thousands of players and an thriving auction house, it is by far the game's most popular mode as players grind towards getting the best version of their all-time favourite players and using them against their friends or other opponents online.

Since there are so many people playing Madden Ultimate Team. The auction house simulates a real economy where prices of the in-game currency, 'coins', are valid determinants of the worth of a card. I have 3 purposes for this repository.

## 1. Muthead Scraper :computer: 
* I've created a [MUTHead.com](https://www.muthead.com/20/players/) scraper to pull all the stats of all the cards in the game. This was done in Python using the Selenium library. 
* The use of Selenium was necessitated since MUTHead dynamically created through Javascript since Madden 20. Unfortunately, even with the aid of the multi-threading library,  it is relatively slow due to MUTHead response times, limitations of Selenium framework, and the high number of cards of released.  
* It is recommended to download the .csv files I commit instead of running the code yourself since it could take up to 2 hours to scrape the stats of the thousands of cards on the site.

## 2. Price Predictor :chart_with_upwards_trend/; :one: :zero: :zero:
I've created a model that predicts the price of a card given its stats. This will aid in indicating which players have a price premium that cannot be explained by statistics, thus allowing me to determine which players are the best value. Furthermore, it should show us what stats are the most valued for each position.

## 3. Position Cluster
* For fun, I want to see if I can create a clustering model that is able to guess a players position based on their stats. There might be some interesting results here that may indicate that some cards would be better suited for positions that they are not currently listed for.
