import mongoose from 'mongoose';

const tokenDataSchema = new mongoose.Schema({
    // Dexscreener's unique pair address (e.g., from PancakeSwap V2 pool)
    pairAddress: { type: String, required: true, unique: true },
    chainId: { type: String, required: true }, // e.g., 'bsc'

    // Metadata about the pair and its base/quote tokens
    baseTokenAddress: { type: String, required: true }, // e.g., BUSD address
    baseTokenSymbol: { type: String, required: true },
    targetTokenAddress: { type: String, required: true }, // The token we are monitoring
    targetTokenSymbol: { type: String, required: true },
    targetTokenName: { type: String, required: true },
    pairName: { type: String, required: true }, // e.g., WBNB/BUSD

    // Historical data arrays
    priceHistory: [{
        price: Number, // Price in USD (from Dexscreener)
        timestamp: { type: Date, default: Date.now }
    }],
    volumeHistory: [{
        volume: Number, // 24-hour volume in USD (from Dexscreener)
        timestamp: { type: Date, default: Date.now }
    }],
    liquidityHistory: [{
        liquidity: Number, // Total liquidity in USD (from Dexscreener)
        timestamp: { type: Date, default: Date.now }
    }],

     // NEW: Historical signals (updated to include predictions)

    signalHistory: [{
        signal: { type: String, enum: ['Buy', 'Sell', 'Hold', 'Error'], required: true },
        currentPrice: Number,
        currentVolume: Number,
        currentLiquidity: Number,
        rsi: String,
        macd: String,
        macdSignal: String,
        macdHistogram: String,
        priceChangeShort: String,
        volumeIncrease: String,
        liquidity: String,
        pumpedRecently: String,
        signalDetails: [String],
        lstmPrediction: mongoose.Schema.Types.Mixed, // New field for LSTM prediction
        xgboostPrediction: mongoose.Schema.Types.Mixed, // New field for XGBoost prediction
        combinedPrediction: mongoose.Schema.Types.Mixed, // New field for combined prediction
        timestamp: { type: Date, default: Date.now }
    }],

    lastUpdated: { type: Date, default: Date.now }
});

// Create indexes for efficient querying
tokenDataSchema.index({ pairAddress: 1 });
tokenDataSchema.index({ targetTokenAddress: 1 });
tokenDataSchema.index({ lastUpdated: 1 }); // Useful for sorting/pruning

tokenDataSchema.index({ 'signalHistory.timestamp': -1 }); // Index for sorting signals by time



const TokenData = mongoose.model('TokenData', tokenDataSchema);

export default TokenData;