// backend/index.js
import express from 'express';
import cors from 'cors';
import * as marketDataService from './services/marketDataService.js';
import * as dataService from './services/dataService.js';
import * as indicatorService from './services/indicatorService.js';
import * as predictionService from './services/predictionService.js';
import config from './config/default.json' assert { type: 'json' };

const app = express();
const PORT = process.env.PORT || config.apiPort;



// Middleware

const allowedOrigins = [
    'http://localhost:5173', // For your local frontend development
    'http://localhost:3000', // If your frontend runs on 3000 for some reason
    'https://pancakeswapfront.vercel.app',
    "https://pancakeswap-signal.vercel.app"
];

const productionFrontendUrl = process.env.FRONTEND_VERCEL_URL;
if (productionFrontendUrl) {
    allowedOrigins.push(productionFrontendUrl);
}

const corsOptions = {
    origin: function (origin, callback) {
        // Allow requests with no origin (like mobile apps or curl requests)
        // or if the origin is in our allowed list.
        if (!origin || allowedOrigins.includes(origin)) {
            callback(null, true);
        } else {
            callback(new Error('Not allowed by CORS'));
        }
    },
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'], // Specify allowed methods
    credentials: true, // If you're sending cookies or authorization headers
    optionsSuccessStatus: 204 // Some legacy browsers (IE11, various SmartTVs) choke on 200
};

app.use(cors(corsOptions));
app.use(express.json());

let currentSignals = [];

async function signalGenerationLoop() {
    console.log(`\n--- [${new Date().toISOString()}] Starting Signal Generation Loop (Inference Only) ---`);
    const allSignals = [];

    for (const tokenConfig of config.monitoredTokens) {
        let signalResult = {
            signal: "Hold",
            pairName: tokenConfig.symbol,
            pairAddress: tokenConfig.address,
            currentPrice: 'N/A',
            currentVolume: 'N/A',
            currentLiquidity: 'N/A',
            rsi: 'N/A',
            macd: 'N/A',
            macdSignal: 'N/A',
            macdHistogram: 'N/A',
            priceChangeShort: 'N/A',
            volumeIncrease: 'N/A',
            liquidity: 'N/A',
            pumpedRecently: 'N/A',
            signalUpdate: { time: 'N/A', price: 'N/A' },
            timeTakenFor1_6_percent: 'N/A',
            signalDetails: [],
            lstmPrediction: 'N/A',
            xgboostPrediction: 'N/A',
            combinedPrediction: 'N/A',
            predictedTime: 'N/A',
            expiryTime: 'N/A'
        };

        try {
            const marketData = await marketDataService.getMarketData(tokenConfig);

            if (!marketData) {
                console.warn(`No market data found for ${tokenConfig.symbol}. Skipping signal generation.`);
                signalResult.signal = "Error";
                signalResult.signalDetails.push(`No market data found.`);
                allSignals.push(signalResult);
                continue;
            }

            const { pairAddress, chainId, baseToken, quoteToken, pairName, currentPrice, currentVolume, currentLiquidity } = marketData;

           
             await dataService.initializeTokenData({
                pairAddress,
                chainId: "bsc",
                baseToken: { address: baseToken.address, symbol: baseToken.symbol },
                quoteToken,
                pairName,
                targetTokenAddress: tokenConfig.address,
                targetTokenSymbol: tokenConfig.symbol,
                targetTokenName: tokenConfig.name,
            });

            // --- IMPORTANT CHANGE HERE ---
            // Only update market data if price, volume, and liquidity are valid numbers
            // if (currentPrice !== null && currentVolume !== null && currentLiquidity !== null) {
            //      await dataService.updateMarketData(pairAddress, currentPrice, currentVolume, currentLiquidity);
            // } else {
            //     console.warn(`Missing or invalid price/volume/liquidity data for ${tokenConfig.symbol}. Skipping market data update.`);
            //     signalResult.signal = "Error";
            //     signalResult.signalDetails.push(`Missing or invalid current market data.`);
            //     allSignals.push(signalResult);
            //     continue; // Skip further processing for this token if core data is missing
            // }

            
            // 3. Store current market data historically in MongoDB
            await dataService.updateMarketData(pairAddress, currentPrice, currentVolume, currentLiquidity);

            if (currentPrice !== null && !isNaN(currentPrice)) {
                signalResult.signalUpdate = {
                    time: new Date().toISOString(), // Current timestamp
                    price: parseFloat(signalResult.currentPrice) // Use the formatted price
                };
            }

            // Generate combined signal based on current and historical data from DB
            signalResult = await indicatorService.generateCombinedSignal(
                pairAddress,
                currentPrice, // Pass the number directly
                currentVolume, // Pass the number directly
                currentLiquidity, // Pass the number directly
                pairName
            );

            // Call the prediction service for INFERENCE ONLY
            const predictionResults = await predictionService.generatePrediction(pairAddress);

            // Add predictions to the signalResult
            signalResult.lstmPrediction = predictionResults.lstmPrediction !== null ? predictionResults.lstmPrediction.toFixed(8) : 'N/A';
            signalResult.xgboostPrediction = predictionResults.xgboostPrediction !== null ? predictionResults.xgboostPrediction.toFixed(8) : 'N/A';
            signalResult.combinedPrediction = predictionResults.combinedPrediction !== null ? predictionResults.combinedPrediction.toFixed(8) : 'N/A';
            if (predictionResults.details && predictionResults.details !== 'Not enough historical data for prediction.') {
                signalResult.signalDetails.push(`Prediction Status: ${predictionResults.details}`);
            }

            // signalResult.predictedTime = predictionResults.predictedTime;
            // signalResult.expiryTime = predictionResults.expiryTime;
            if (signalResult.lstmPrediction !== null && signalResult.lstmPrediction !== 'N/A' ||
                    signalResult.combinedPrediction !== null && signalResult.combinedPrediction !== 'N/A') {
                    signalResult.predictedTime = predictionResults.predictedTime;
                    signalResult.expiryTime = predictionResults.expiryTime;
            }else{
                signalResult.predictedTime = 'N/A';
                signalResult.expiryTime =  'N/A';
            }

            allSignals.push(signalResult);
            await dataService.updateSignalHistory(pairAddress, signalResult);

        } catch (error) {
            console.error(`Error processing ${tokenConfig.symbol}:`, error);
            signalResult.signal = "Error";
            signalResult.signalDetails.push(`An unexpected error occurred: ${error.message}`);
            allSignals.push(signalResult);
        }
    }

    currentSignals = allSignals;
    console.log(`--- Signal Generation Loop Finished. ${allSignals.length} signals processed. ---`);
}

// API Endpoint
app.get('/api/signals', (req, res) => {
    // res.json(currentSignals);
    if (currentSignals.length === 0) {
        // If no signals but loop is running, implies initial generation is in progress
        res.status(202).json({ message: "Signals are being generated. Please wait.", status: "generating" });
    }
    else {
        res.json(currentSignals);
    }
});

// Start the signal generator and API server
// async function startSignalGeneratorAndApi() {
//     console.log('Starting signal generator and API server...');

//     await dataService.connectDb(); // Initial database connection

//     // Load models at startup. If no saved models, they will be null.
//     await predictionService.loadModels();

//     // Initial training for models if they were not loaded (e.g., first run)
//     // Add a slight delay to ensure DB is fully ready and initial data might be fetched.
//     setTimeout(async () => {
//         if (!predictionService.hasModelsLoaded()) {
//             console.log("Models not loaded at startup, performing initial training...");
//             await predictionService.retrainModels(); // Initial training after a delay
//         }
//     }, config.initialModelTrainingDelayMs || 5000); // Default to 5 seconds if not in config

//     // Schedule periodic model retraining
//     setInterval(predictionService.retrainModels, config.modelRetrainIntervalMs);

//     // Run first signal generation immediately
//     await signalGenerationLoop();

//     // Schedule subsequent signal generation runs
//     setInterval(signalGenerationLoop, config.refreshIntervalMs);

//     app.listen(PORT, () => {
//         console.log(`API Server listening on port ${PORT}`);
//         console.log(`Access signals at http://localhost:${PORT}/api/signals`);
//         console.log(`Remember to also start your React frontend in a separate terminal`);
//     }).on('error', (err) => {
//         console.error('Failed to start API server:', err.message);
//         if (err.code === 'EADDRINUSE') {
//             console.error(`Port ${PORT} is already in use. Please close the other application or choose a different port.`);
//         }
//         process.exit(1);
//     });
// }

async function startSignalGeneratorAndApi() {
    console.log('Starting signal generator and API server...');

    // 1. Initial database connection (keep this awaited)
    await dataService.connectDb();

    // 2. Start Express API server first, so it's immediately available
    app.listen(PORT, () => {
        console.log(`API Server listening on port ${PORT}`);
        console.log(`Access signals at http://localhost:${PORT}/api/signals`);
        console.log(`Remember to also start your React frontend in a separate terminal`);
    }).on('error', (err) => {
        console.error('Failed to start API server:', err.message);
        if (err.code === 'EADDRINUSE') {
            console.error(`Port ${PORT} is already in use. Please close the other application or choose a different port.`);
        }
        process.exit(1); // Exit if server cannot start
    });

    // 3. Load prediction models (keep this awaited)
    await predictionService.loadModels();

    // 4. Initial training for models if they were not loaded (e.g., first run)
    // Add a slight delay to ensure DB is fully ready and initial data might be fetched.
    setTimeout(async () => {
        if (!predictionService.hasModelsLoaded()) {
            console.log("Models not loaded at startup, performing initial training...");
            await predictionService.retrainModels(); // Initial training after a delay
        }
    }, config.initialModelTrainingDelayMs || 5000); // Default to 5 seconds if not in config

    // 5. Schedule periodic model retraining
    setInterval(predictionService.retrainModels, config.modelRetrainIntervalMs);

    // 6. Initiate the first signal generation loop.
    // Do NOT await this call. Let it run in the background.
    console.log('Initiating first signal generation. Signals will be available soon...');
    signalGenerationLoop();

    // 7. Schedule subsequent signal generation runs
    setInterval(signalGenerationLoop, config.refreshIntervalMs);
}

startSignalGeneratorAndApi();