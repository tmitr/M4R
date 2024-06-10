# Optimal LP and ImPL Hedging Project

## Top Level Idea:

1. Provide liquidity on Uniswap v3 within a range
2. Hedge ImPL using options on Centralised exchange

*Objectives*:
- model the liquidity dynamics (perhaps using traditional order book techniques)
- find the optimal hedge / LP interval (jointly?) with respect to some utility
- thereby exploiting dynamics across exchanges

*First POA*:
- Understand the liquidity with the 'histogram' approach
- Explore the data, get familiar
- Draft a first roadmap. Need to know
    1. Understand the problem/data -> empirical study
    2. Understand how to model dynamics (traditional methods + empirical observations)
    3. How to setup utility/objective
    4. How to solve (dynamic programming / RL)

## Data:

There exists a [README.md](README.md) file in the data folder, provided by Adrian. Any big data files, must be stored in the [.gitattributes](.gitattributes) file to deal with Git large file storage (LFS).

## Repo Structure

- *data*: storing any files including Uniswap v3 data or cleaned versions
- *notebooks*: for playing around with the data and exploratory analysis, feel free to create your own folder if you think it will get messy
- *src*: for any universal functionality / python files that can be imported into notebooks (or other python files)
- *tests*: for any rigorous functionality to test and check
- *ideas/meetings*: for any blueprints and general important documentation of ideas, breakthroughs or important info that should be shared (to be kept in a nice readable format)
