// services/indicatorService.js
import TI from 'technicalindicators';
import config from '../config/default.json' assert { type: 'json' };
import * as dataService from './dataService.js'; // Import dataService

/**
 * Calculates the Relative Strength Index (RSI).
 * @param {Array<object>} priceHistory - Array of objects [{ price: number, timestamp: Date }]
 * @param {number} period - RSI period.
 * @returns {number|null} Latest RSI value or null if not enough data.
 */
function calculateRSI(priceHistory, period = config.indicatorPeriodRSI) {
    if (!priceHistory || priceHistory.length < period) {
        return null;
    }
    const prices = priceHistory.map(d => d.price);
    const rsiResult = TI.RSI.calculate({ values: prices, period });
    return rsiResult.length > 0 ? rsiResult[rsiResult.length - 1] : null;
}

/**
 * Calculates the Moving Average Convergence Divergence (MACD).
 * @param {Array<object>} priceHistory - Array of objects [{ price: number, timestamp: Date }]
 * @returns {object|null} Latest MACD values (MACD, Signal, Histogram) or null.
 */
function calculateMACD(priceHistory) {
    if (!priceHistory || priceHistory.length < config.macdSlowPeriod) { // MACD needs more data than RSI typically
        return null;
    }
    const prices = priceHistory.map(d => d.price);
    const macdResult = TI.MACD.calculate({
        values: prices,
        fastPeriod: config.macdFastPeriod,
        slowPeriod: config.macdSlowPeriod,
        signalPeriod: config.macdSignalPeriod,
        SimpleMAOscillator: false,
        SimpleMASignal: false
    });
    return macdResult.length > 0 ? macdResult[macdResult.length - 1] : null;
}

/**
 * Checks for rapid short-term price increase.
 * @param {Array<object>} priceHistory - Array of objects [{ price: number, timestamp: Date }]
 * @param {number} currentPrice - Current price of the token.
 * @returns {object} Result, percentage change, and reason.
 */
function isPriceRisingRapidly(priceHistory, currentPrice) {
    
    const lookbackMinutes = config.priceChangeLookbackMinutesShort;
    const threshold = config.priceChangeThresholdShort; // e.g., 0.01 for 1%
    const now = Date.now();

    const recentPrices = priceHistory.filter(d => (now - d.timestamp.getTime()) <= lookbackMinutes * 60 * 1000);

    if (recentPrices.length < 2) {
        return { result: false, change: 0, reason: `Not enough data for ${lookbackMinutes} min price trend.` };
    }

    const oldestPricePoint = recentPrices[0];
    const oldestPrice = oldestPricePoint.price;
    if (oldestPrice === 0) { // Avoid division by zero
        return { result: false, change: 0, reason: `Oldest price is zero for ${lookbackMinutes} min trend.` };
    }

    const change = (currentPrice - oldestPrice) / oldestPrice;
    const result = change >= threshold;
    const reason = result
        ? `Price increased by ${((change * 100).toFixed(2))}% in last ${lookbackMinutes} min (>=${(threshold * 100).toFixed(2)}% required).`
        : `Price only increased by ${((change * 100).toFixed(2))}% in last ${lookbackMinutes} min (<${(threshold * 100).toFixed(2)}% required).`;

    return { result, change, reason };
}

/**
 * Checks if 24-hour volume is significantly increasing compared to an average.
 * Dexscreener gives us a 24h volume. We can compare current 24h volume to previous average 24h volume values.
 * @param {Array<object>} volumeHistory - Array of objects [{ volume: number, timestamp: Date }]
 * @param {number} currentVolume - Current 24h volume of the token.
 * @returns {object} Result, current volume, average volume, and reason.
 */
function isVolumeIncreasing(volumeHistory, currentVolume) {
    const currentVolumes = Number(currentVolume);
    const lookbackMinutes = config.volumeLookbackMinutes;
    const increaseFactor = config.volumeIncreaseFactor; // e.g., 0.2 for 20%
    const now = Date.now();

    const recentVolumes = volumeHistory.filter(d => (now - d.timestamp.getTime()) <= lookbackMinutes * 60 * 1000);

    if (recentVolumes.length < 5) { // Need a few points to calculate a meaningful average
        return { result: false, current: currentVolume, average: 0, reason: `Not enough data for volume trend over ${lookbackMinutes} min.` };
    }

    // Exclude the most recent volume from average calculation to avoid skewing
    const volumesForAverage = recentVolumes.slice(0, -1).map(d => d.volume); 
    const averageVolume = volumesForAverage.reduce((sum, vol) => sum + vol, 0) / volumesForAverage.length;

    const result = currentVolumes >= averageVolume * (1 + increaseFactor);
    const reason = result
        ? `Current volume (${currentVolumes.toFixed(2)}) is ${(((currentVolumes / averageVolume - 1) * 100).toFixed(2))}% higher than average (${averageVolume.toFixed(2)}) over ${lookbackMinutes} min.`
        : `Current volume (${currentVolumes.toFixed(2)}) is not significantly higher than average (${averageVolume.toFixed(2)}) over ${lookbackMinutes} min.`;


     console.log(result);

    return { result, current: currentVolume, average: averageVolume, reason };
}

/**
 * Checks if liquidity is above a minimum threshold.
 * @param {number} currentLiquidity - Current total liquidity in USD.
 * @returns {object} Result and reason.
 */
function checkLiquidityStatus(currentLiquidity) {


    // console.log("the current liquidity is", currentLiquidity);

    const threshold = config.liquidityThreshold;
    const result = Number(currentLiquidity) >= Number(config.liquidityThreshold);
    const reason = result
        ? `Liquidity (${Number(currentLiquidity).toFixed(2)}) is above threshold (${threshold}).`
        : `Liquidity (${Number(currentLiquidity).toFixed(2)}) is below threshold (${threshold}).`;

        // console.log(reason, result, "the threshold is", threshold, "the current liquidity is", currentLiquidity);
    return { result, reason };
}

/**
 * Checks if the price has pumped recently (within a configurable lookback period).
 * This condition should generally be FALSE for a "buy" signal, indicating we want to avoid FOMO.
 * @param {Array<object>} priceHistory - Array of objects [{ price: number, timestamp: Date }]
 * @param {number} currentPrice - Current price of the token.
 * @returns {object} Result (true if pumped, false otherwise) and reason.
 */
function hasPumpedRecently(priceHistory, currentPrice) {
    const lookbackHours = config.priceChangeLookbackHoursPumped;
    const threshold = config.priceChangeThresholdPumped; // e.g., 0.10 for 10% pump
    const now = Date.now();

    const recentPrices = priceHistory.filter(d => (now - d.timestamp.getTime()) <= lookbackHours * 60 * 60 * 1000);

    if (recentPrices.length < 2) {
        return { result: false, change: 0, reason: `Not enough data for ${lookbackHours} hr pump check.` };
    }

    const oldestPricePoint = recentPrices[0];
    const oldestPrice = oldestPricePoint.price;
    if (oldestPrice === 0) {
        return { result: false, change: 0, reason: `Oldest price is zero for ${lookbackHours} hr pump check.` };
    }

    const change = (currentPrice - oldestPrice) / oldestPrice;
    const result = change >= threshold;
    const reason = result
        ? `Price increased by ${((change * 100).toFixed(2))}% in last ${lookbackHours} hr (>=${(threshold * 100).toFixed(2)}% threshold).`
        : `Price only increased by ${((change * 100).toFixed(2))}% in last ${lookbackHours} hr (<${(threshold * 100).toFixed(2)}% threshold).`;

    return { result, change, reason };
}


/**
 * Generates a combined buy/sell/hold signal based on multiple indicators.
 * @param {string} pairAddress - The unique address of the token pair.
 * @param {number} currentPrice - Current price of the token.
 * @param {number} currentVolume - Current 24-hour volume of the token.
 * @param {number} currentLiquidity - Current total liquidity of the token.
 * @param {string} pairName - The name of the token pair for logging.
 * @returns {object} An object containing the signal, current data, and indicator values.
 */
export async function generateCombinedSignal(pairAddress, currentPrice, currentVolume, currentLiquidity, pairName) {

    console.log("currentLiquidity", currentLiquidity);
    let signal = "Hold";
    const signalDetails = [];

    // Fetch full history from DB
    const tokenData = await dataService.getTokenData(pairAddress);
    if (!tokenData) {
        signalDetails.push("No historical data found for indicators.");
        return { signal: "Hold", pairName, currentPrice, currentVolume, currentLiquidity, signalDetails };
    }

    const priceHistory = tokenData.priceHistory;
    const volumeHistory = tokenData.volumeHistory;
    const liquidityHistory = tokenData.liquidityHistory;


    // 1. RSI Calculation & Condition
    const rsi = calculateRSI(priceHistory);
    const rsiCondition = rsi !== null && rsi <= config.rsiOversold;
    signalDetails.push(`RSI (${rsi !== null ? rsi.toFixed(2) : 'N/A'}): ${rsiCondition ? '✅ Oversold' : '❌ Not Oversold'}`);

    // 2. MACD Calculation & Condition
    const macd = calculateMACD(priceHistory);
    // Bullish crossover: MACD line crosses above Signal line
    // Also consider if both are positive or near zero for stronger signal
    const macdCondition = macd !== null && macd.MACD > macd.signal;
    signalDetails.push(`MACD (${macd !== null ? macd.MACD.toFixed(4) : 'N/A'} vs Signal ${macd !== null ? macd.signal.toFixed(4) : 'N/A'}): ${macdCondition ? '✅ Bullish Crossover' : '❌ No Bullish Crossover'}`);


    // 3. Price Trend (Short-term rapid increase check)
    const priceTrend = isPriceRisingRapidly(priceHistory, currentPrice);
    signalDetails.push(`Short-term Price Trend (Last ${config.priceChangeLookbackMinutesShort} min): ${priceTrend.reason} (${priceTrend.result ? '❌' : '✅'}- Must NOT be rapidly rising)`);


    // 4. Volume Trend
    const volumeTrend = isVolumeIncreasing(volumeHistory, currentVolume);
    signalDetails.push(`Volume Trend (Last ${config.volumeLookbackMinutes} min): ${volumeTrend.reason} (${volumeTrend.result ? '✅' : '❌'})`);


    // 5. Liquidity Check
    const liquidityStatus = checkLiquidityStatus(currentLiquidity);
    signalDetails.push(`Liquidity Status: ${liquidityStatus.reason} (${liquidityStatus.result ? '✅' : '❌'})`);

    const pumpedStatus = hasPumpedRecently(priceHistory, currentPrice);
    signalDetails.push(`Recently Pumped (Last ${config.priceChangeLookbackHoursPumped} hr): ${pumpedStatus.reason} (${!pumpedStatus.result ? '✅' : '❌'}- Must NOT have pumped)`);


    // Combined Buy Logic
    const canBuy =
        rsiCondition &&
        macdCondition &&
        !priceTrend.result && // Must NOT be rapidly rising (anti-pump)
        volumeTrend.result &&
        liquidityStatus.result &&
        !pumpedStatus.result; // Must NOT have pumped recently (anti-FOMO)

    if (canBuy) {
        signal = "Buy";
    }

    return {
        signal,
        pairName,
        currentPrice: parseFloat(currentPrice).toFixed(config.priceDecimals), // Format for display
        currentVolume: parseFloat(currentVolume).toFixed(2),
        currentLiquidity: parseFloat(currentLiquidity).toFixed(2),
        rsi: rsi !== null ? rsi.toFixed(2) : 'N/A',
        macd: macd !== null ? macd.MACD.toFixed(4) : 'N/A',
        macdSignal: macd !== null ? macd.signal.toFixed(4) : 'N/A',
        macdHistogram: macd !== null ? macd.histogram.toFixed(4) : 'N/A',
        priceChangeShort: (priceTrend.change * 100).toFixed(2),
        volumeIncrease: (volumeTrend.current / volumeTrend.average - 1) * 100 >= 0 ? ((volumeTrend.current / volumeTrend.average - 1) * 100).toFixed(2) : 'N/A',
        liquidity: Number(currentLiquidity).toFixed(2),
        signalDetails // Pass the array of detailed reasons
    };
}