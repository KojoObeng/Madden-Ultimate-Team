# Madden Ultimate Team

## What is Madden Ultimate Team?

Madden Ultimate Team is a popular mode in EA's Madden NFL video game series. Every NFL athlete has multiple cards of their themself, with their stats generally improving as the year progresses. Cards can be bought for in-game currency on the auction house, or real money, and then used to play against other players across the world. With hundreds of thousands of players and an thriving auction house, it is by far the game's most popular mode as players grind towards getting the best version of their favourite players of all-time and using them against their friends or other opponents online.

Since there are so many people playing Madden Ultimate Team. The auction house simulates a real economy where prices of the in-game currency, 'coins', are valid determinants of the worth of a card.

#Road Map

## Muthead Scraper
I've created a [MUTHead](https://www.muthead.com/20/players/) scraper to pull all the stats of all the cards in the game. This was done in Python and Selenium. The use of Selenium was necessitated since MUTHead dynamically created through Javascript since Madden 20. Unfortunately, it is very slow due to MUTHead response times and the limitations of Selenium framework. It is recommended to download the .csv files I commit instead of running the code yourself since it could take up to 2 hours to scrape the stats of the thousands of cards on the site.

## Price Predictor
I want to create
