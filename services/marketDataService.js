import { request, gql } from 'graphql-request';
import axios from 'axios';
import config from '../config/default.json' assert { type: 'json' };
import Moralis from 'moralis';

// Dexscreener Config
const DEXSCREENER_API_SEARCH_BASE_URL = config.dexscreenerApiSearchBaseUrl;

const quoteSymbol = config.preferredQuoteTokenSymbols.map(s => s.toUpperCase());

const THEGRAPH_API_KEY = process.env.SUBGRAPH_API_KEY;
const PANCAKESWAP_V3_SUBGRAPH_URL = 'https://gateway.thegraph.com/api/subgraphs/id/Hv1GncLY5docZoGtXjo4kwbTvxm3MAhVZqBZE4sUT9eZ';

const SUBGRAPH_HEADERS = {
    'Authorization': `Bearer ${THEGRAPH_API_KEY}`,
};

const QUOTE_TOKEN_ADDRESSES = Object.values(config.quoteTokenMap).map(address => address.toLowerCase());
const HISTORICAL_DATA_DAYS = 60; // For RSI/MACD calculations

// Moralis API Key from environment variables
const MORALIS_API_KEY = process.env.MORALIS_API_KEY;

// Initialize Moralis SDK once at the module level
async function initializeMoralis() {
    if (MORALIS_API_KEY && !Moralis.Core.isStarted) {
        try {
            await Moralis.start({
                apiKey: MORALIS_API_KEY
            });
            console.log('Moralis SDK initialized successfully.');
        } catch (e) {
            console.error('Failed to initialize Moralis SDK:', e);
        }
    }
}
// Call initialization function immediately
initializeMoralis();

/**
 * Subgraph query to get token and its daily historical data for V3,
 * and separate query for relevant pools.
 */
const GET_TOKEN_AND_POOL_DATA_V3 = gql`
    query GetTokenAndPoolDataV3($tokenId: String!, $quoteToken0: String!, $quoteToken1: String!) {
        token(id: $tokenId) {
            id
            name
            symbol
            decimals
            derivedUSD # Current price in USD based on its pools
            volumeUSD # Total accumulated volume across all pools for this token
            totalValueLockedUSD # Total accumulated TVL across all pools for this token
            txCount # Total transactions for this token

            tokenDayData(orderBy: date, orderDirection: desc, first: ${HISTORICAL_DATA_DAYS}) {
                date
                priceUSD
                volumeUSD # Daily volume in USD for this token
                totalValueLockedUSD # Daily TVL in USD for this token
            }
        }

        # Query for pools specifically involving our target token and common quote tokens
        # We need to explicitly filter based on token0 or token1 being our target token
        # and the other token being a quote token (WBNB or BUSD).
        pools(
            first: 10,
            orderBy: totalValueLockedUSD,
            orderDirection: desc,
            where: {
                or: [
                    {
                        token0_: { id: $tokenId },
                        token1_: { id_in: [$quoteToken0, $quoteToken1] }
                    },
                    {
                        token1_: { id: $tokenId },
                        token0_: { id_in: [$quoteToken0, $quoteToken1] }
                    }
                ]
            }
        ) {
            id
            token0 { id symbol decimals }
            token1 { id symbol decimals }
            volumeUSD
            totalValueLockedUSD
            token0Price
            token1Price
            feesUSD
        }
    }
`;


function safeParseFloat(value) {
    if (value === undefined || value === null || value === '') {
        return null;
    }
    const parsed = parseFloat(value);
    return isNaN(parsed) ? null : parsed;
}

// Utility function for sleeping
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Retry mechanism with exponential backoff
async function retry(fn, retries = 3, delay = 1000) {
    for (let i = 0; i < retries; i++) {
        try {
            return await fn();
        } catch (error) {
            if (error.response && error.response.status === 429) {
                console.warn(`Rate limit hit (429). Retrying in ${delay / 1000}s... (Attempt ${i + 1}/${retries})`);
                await sleep(delay);
                delay *= 2;
            } else {
                throw error;
            }
        }
    }
    throw new Error(`Failed after ${retries} retries due to persistent rate limiting.`);
}

export async function getMarketData(tokenConfig) {
    const { address: targetTokenAddress, symbol: targetTokenSymbol, name: targetTokenName } = tokenConfig;
    const lowerCaseTokenAddress = targetTokenAddress.toLowerCase();

    let marketData = null;
    let historicalPrices = [];

    let currentPriceFinal = null;
    let currentVolumeFinal = null;
    let currentLiquidityFinal = null;
    let pairAddressUsed = lowerCaseTokenAddress;

    // --- 1. Try to fetch from PancakeSwap V3 Subgraph ---
    console.log(`Attempting to fetch market data for ${targetTokenSymbol} from PancakeSwap V3 Subgraph...`);
    try {
        if (!THEGRAPH_API_KEY) {
            console.warn("THEGRAPH_API_KEY is not set. Skipping Subgraph query.");
            throw new Error("THEGRAPH_API_KEY not configured.");
        }

        const variables = {
            tokenId: lowerCaseTokenAddress,
            quoteToken0: config.quoteTokenMap.WBNB.toLowerCase(),
            quoteToken1: config.quoteTokenMap.BUSD.toLowerCase()
        };

        const subgraphResponse = await request(
            PANCAKESWAP_V3_SUBGRAPH_URL,
            GET_TOKEN_AND_POOL_DATA_V3,
            variables,
            SUBGRAPH_HEADERS
        );

        const token = subgraphResponse.token;
        const pools = subgraphResponse.pools;

        if (token) {
            console.log(`Token data found for ${targetTokenSymbol} on V3 Subgraph.`);

            let tokenPriceUsdFromBUSD = null;
            for (let i = 0; i < pools.length; i++) {
                const pool = pools[i];
                const formattedPrice = Number(pool.token1Price).toFixed(6);
                if (pool.token1.symbol === 'BUSD' && parseFloat(formattedPrice) < 10000) {
                    tokenPriceUsdFromBUSD = Number(pool.token1Price).toFixed(6);
                    break;
                }
            }

            if (token.derivedUSD && Number(token.derivedUSD).toFixed(6) < 100000) {
                currentPriceFinal = safeParseFloat(Number(token.derivedUSD).toFixed(6));
            } else if (tokenPriceUsdFromBUSD !== null) {
                currentPriceFinal = safeParseFloat(tokenPriceUsdFromBUSD);
            }

            // --- 1.1. Specific currentPrice Fallback: Subgraph Price -> Moralis -> Dexscreener ---
            if (currentPriceFinal === null || isNaN(currentPriceFinal)) {
                console.warn(`Subgraph price for ${targetTokenSymbol} is not valid. Attempting Moralis fallback for currentPrice.`);

                // Try Moralis first for currentPriceFinal
                try {
                    if (MORALIS_API_KEY && Moralis.Core.isStarted) {
                        const moralisResponse = await Moralis.EvmApi.token.getTokenPrice({
                            "chain": config.targetChainIds,
                            "address": lowerCaseTokenAddress,
                            "include": "percent_change"
                        });

                        if (moralisResponse && moralisResponse.raw && moralisResponse.raw.usdPrice) {
                            currentPriceFinal = safeParseFloat(moralisResponse.raw.usdPrice);
                            console.log("price fetched from moralis", currentPriceFinal)
                            console.log(`Successfully fetched current price for ${targetTokenSymbol} from Moralis (Subgraph fallback).`);
                        } else {
                            console.warn(`No price data found for ${targetTokenSymbol} on Moralis during currentPrice fallback.`);
                        }
                    } else {
                        console.warn("Moralis SDK not initialized or API Key missing. Cannot use Moralis for currentPrice fallback.");
                    }
                } catch (moralisError) {
                    console.error(`Moralis current price fetch failed for ${targetTokenSymbol} (Subgraph fallback):`, moralisError.message);
                }

                // If Moralis failed for currentPrice, then try Dexscreener
                if (currentPriceFinal === null || isNaN(currentPriceFinal)) {
                    console.warn(`Moralis currentPrice fallback failed. Attempting Dexscreener fallback for currentPrice.`);
                    const queryString = `${targetTokenSymbol}/${quoteSymbol}`;
                    const dexscreenerApiUrl = `${DEXSCREENER_API_SEARCH_BASE_URL}?q=${queryString}`;

                    try {
                        const response = await retry(async () => {
                            return await axios.get(dexscreenerApiUrl, { timeout: config.dexscreenerTimeoutMs });
                        }, 3, 1000);

                        if (response.data && response.data.pairs && response.data.pairs.length > 0) {
                            const preferredQuoteTokens = Object.values(config.quoteTokenMap).map(address => address.toLowerCase());
                            const relevantPairs = response.data.pairs.filter(p =>
                                p.chainId === config.chainId &&
                                (preferredQuoteTokens.includes(p.baseToken.address.toLowerCase()) ||
                                 preferredQuoteTokens.includes(p.quoteToken.address.toLowerCase()))
                            );
                            let pair = null;
                            if (relevantPairs.length > 0) {
                                pair = relevantPairs.sort((a, b) => b.liquidity.usd - a.liquidity.usd)[0];
                            } else {
                                pair = response.data.pairs.sort((a, b) => b.liquidity.usd - a.liquidity.usd)[0];
                            }
                            if (pair && pair.priceUsd) {
                                currentPriceFinal = safeParseFloat(Number(pair.priceUsd));
                                // currentPriceFinal = safeParseFloat(parseFloat(pair.priceUsd).toFixed(config.priceDecimals));
                                console.log(`Successfully fetched current price for ${targetTokenSymbol} from Dexscreener (Subgraph fallback).`);
                            }
                        }
                    } catch (dexscreenerError) {
                        console.error(`Dexscreener initial price fetch failed after retries for ${targetTokenSymbol} (Subgraph fallback):`, dexscreenerError.message);
                    }
                }
            }

            currentVolumeFinal = safeParseFloat(Number(parseFloat(token.volumeUSD)).toFixed(2));
            currentLiquidityFinal = safeParseFloat(Number(parseFloat(token.totalValueLockedUSD)).toFixed(2));

            let mainPool = null;
            if (pools && pools.length > 0) {
                const relevantPools = pools.filter(p =>
                    (p.token0.id === lowerCaseTokenAddress && QUOTE_TOKEN_ADDRESSES.includes(p.token1.id)) ||
                    (p.token1.id === lowerCaseTokenAddress && QUOTE_TOKEN_ADDRESSES.includes(p.token0.id))
                );

                if (relevantPools.length > 0) {
                    mainPool = relevantPools.sort((a, b) => parseFloat(b.totalValueLockedUSD) - parseFloat(a.totalValueLockedUSD))[0];
                } else {
                    mainPool = pools.sort((a, b) => parseFloat(b.totalValueLockedUSD) - parseFloat(a.totalValueLockedUSD))[0];
                }

                if (mainPool) {
                    console.log(`Using data from most liquid pool for ${targetTokenSymbol}: ${mainPool.id}`);
                    currentLiquidityFinal = safeParseFloat(parseFloat(mainPool.totalValueLockedUSD).toFixed(2));
                    currentVolumeFinal = safeParseFloat(parseFloat(mainPool.volumeUSD).toFixed(2));
                    pairAddressUsed = mainPool.id;

                    let poolSpecificPrice = null;
                    if (mainPool.token0.id === lowerCaseTokenAddress) {
                        poolSpecificPrice = safeParseFloat(parseFloat(mainPool.token0Price).toFixed(config.priceDecimals));
                    } else if (mainPool.token1.id === lowerCaseTokenAddress) {
                        poolSpecificPrice = safeParseFloat(parseFloat(mainPool.token1Price).toFixed(config.priceDecimals));
                    }
                    // if (poolSpecificPrice !== null && !isNaN(poolSpecificPrice)) {
                    //     currentPriceFinal = poolSpecificPrice;
                    // }
                }
            }

            if (token.tokenDayData && token.tokenDayData.length > 0) {
                historicalPrices = token.tokenDayData
                    .map(d => ({ date: d.date, price: parseFloat(d.priceUSD) }))
                    .sort((a, b) => a.date - b.date);
            }

            if (historicalPrices.length < config.minHistoricalDataPoints) {
                console.warn(`Insufficient historical data from Subgraph for ${targetTokenSymbol} (${historicalPrices.length} days, needed ${config.minHistoricalDataPoints}).`);
                // Do not throw error here, allow overall fallbacks to try other sources for full data
            }

            marketData = {
                pairAddress: pairAddressUsed,
                chainId: config.targetChainId,
                pairName: `${targetTokenSymbol}/${config.baseCurrencySymbol}`,
                baseToken: { address: targetTokenAddress, symbol: targetTokenSymbol },
                quoteToken: { address: config.quoteTokenMap.WBNB, symbol: "WBNB" },
                currentPrice: currentPriceFinal,
                currentVolume: currentVolumeFinal,
                currentLiquidity: currentLiquidityFinal,
                historicalPrices: historicalPrices,
            };

            console.log("the current price for ", targetTokenSymbol, marketData.currentPrice);   
            // console.log("the market data for ", targetTokenSymbol, marketData);        

            return marketData;
        } else {
              console.warn(`No token data found for ${targetTokenSymbol} on V3 Subgraph.`);
        }
    } catch (subgraphError) {
        console.error(`Subgraph query failed for ${targetTokenSymbol}:`, subgraphError.message);
        if (subgraphError.response?.errors) {
            console.error("Subgraph GraphQL Errors:", subgraphError.response.errors);
        }
        if (subgraphError.message.includes("401") || subgraphError.message.includes("403") || subgraphError.message.includes("Unauthorized")) {
             throw { isAuthError: true, message: "Subgraph API Key Unauthorized" };
        }
    }


    // --- 2. Overall Fallback: Try Moralis if Subgraph failed or had insufficient data ---
    if (marketData === null) {
        console.log(`Attempting to fetch price and basic market data for ${targetTokenSymbol} from Moralis (overall fallback)...`);
        try {
            if (!MORALIS_API_KEY || !Moralis.Core.isStarted) {
                console.warn("Moralis SDK not initialized or API Key missing. Skipping Moralis query.");
            } else {
                const moralisResponse = await Moralis.EvmApi.token.getTokenPrice({
                    "chain": config.targetChainIds,
                    "address": lowerCaseTokenAddress,
                    "include": "percent_change"
                });

                if (moralisResponse && moralisResponse.raw && moralisResponse.raw.usdPrice) {
                    currentPriceFinal = safeParseFloat(moralisResponse.raw.usdPrice);
                    currentVolumeFinal = null;
                    currentVolumeFinal = safeParseFloat(moralisResponse.raw.pairTotalLiquidityUsd);
                    historicalPrices = [];

                    marketData = {
                        pairAddress: lowerCaseTokenAddress,
                        chainId: config.targetChainId,
                        pairName: `${targetTokenSymbol}/USD`,
                        baseToken: { address: targetTokenAddress, symbol: targetTokenSymbol },
                        quoteToken: { address: config.baseCurrencyAddress, symbol: config.baseCurrencySymbol },
                        currentPrice: currentPriceFinal,
                        currentVolume: currentVolumeFinal,
                        currentLiquidity: currentLiquidityFinal,
                        historicalPrices: historicalPrices,
                    };
                    console.log(`Successfully fetched price for ${targetTokenSymbol} from Moralis.`);
                } else {
                    console.warn(`No price data found for ${targetTokenSymbol} on Moralis.`);
                }
            }
        } catch (moralisError) {
            console.error(`Moralis fetch failed for ${targetTokenSymbol}:`, moralisError.message);
        }
    }

    // --- 3. Overall Fallback: Try Dexscreener if Moralis also failed (or was skipped) ---
    if (marketData === null) {
        console.log(`Attempting to fetch market data for ${targetTokenSymbol} from Dexscreener (overall fallback)...`);
        try {
            const queryString = `${targetTokenSymbol}/${quoteSymbol}`;
            const dexscreenerApiUrl = `${DEXSCREENER_API_SEARCH_BASE_URL}?q=${queryString}`;

            const response = await retry(async () => {
                return await axios.get(dexscreenerApiUrl, { timeout: config.dexscreenerTimeoutMs });
            }, 3, 1000);

            if (!response.data || !response.data.pairs || response.data.pairs.length === 0) {
                console.warn(`No pairs found for ${targetTokenSymbol} on Dexscreener.`);
            } else {
                const preferredQuoteTokens = Object.values(config.quoteTokenMap).map(address => address.toLowerCase());
                const relevantPairs = response.data.pairs.filter(p =>
                    p.chainId === config.chainId &&
                    (preferredQuoteTokens.includes(p.baseToken.address.toLowerCase()) ||
                     preferredQuoteTokens.includes(p.quoteToken.address.toLowerCase()))
                );

                let pair = null;
                if (relevantPairs.length > 0) {
                    pair = relevantPairs.sort((a, b) => b.liquidity.usd - a.liquidity.usd)[0];
                } else {
                    console.warn(`No preferred quote token pairs found for ${targetTokenSymbol} on Dexscreener. Taking most liquid pair.`);
                    pair = response.data.pairs.sort((a, b) => b.liquidity.usd - a.liquidity.usd)[0];
                }

                if (pair) {
                    currentPriceFinal = pair.priceUsd ? safeParseFloat(Number(parseFloat(pair.priceUsd)).toFixed(8)) : null;
                    currentVolumeFinal = pair.volume && pair.volume.h24 ? safeParseFloat(parseFloat(pair.volume.h24).toFixed(2)) : null;
                    currentLiquidityFinal = pair.liquidity && pair.liquidity.usd ? safeParseFloat(parseFloat(pair.liquidity.usd).toFixed(2)) : null;
                    historicalPrices = [];

                    marketData = {
                        pairAddress: pair.pairAddress,
                        chainId: pair.chainId,
                        pairName: pair?.baseToken?.symbol + '/' + pair?.quoteToken?.symbol,
                        baseToken: { address: pair.baseToken.address, symbol: pair.baseToken.symbol },
                        quoteToken: { address: pair.quoteToken.address, symbol: pair.quoteToken.symbol },
                        currentPrice: currentPriceFinal,
                        currentVolume: currentVolumeFinal,
                        currentLiquidity: currentLiquidityFinal,
                        historicalPrices: historicalPrices,
                    };
                    console.log(`Successfully fetched data for ${targetTokenSymbol} from Dexscreener.`);
                } else {
                    console.warn(`No suitable pair found for ${targetTokenSymbol} on Dexscreener.`);
                }
            }
        } catch (error) {
            console.error(`Dexscreener fetch failed for ${targetTokenSymbol}:`, error.message);
        }
    }
    
    console.log("the current price for ", targetTokenSymbol, marketData.currentPrice)

    return marketData;
}