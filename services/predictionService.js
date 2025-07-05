import * as tf from "@tensorflow/tfjs-node";
import * as dataService from './dataService.js';
import TI from 'technicalindicators';
import config from '../config/default.json' assert { type: 'json' };
import XGBoost from 'ml-xgboost'; // Import XGBoost
import * as fs from 'fs/promises'; // For file system operations (saving/loading)
import * as path from 'path';




//Global variables to store models
let lstmModel = null;
let xgboostModel = null;


// File paths for saving/loading models
const LSTM_MODEL_PATH = 'file://./models/lstm_model';
const XGBOOST_MODEL_PATH = './models/xgboost_model.json';


//a function to create the Lstm Dataset
function createLstmDataset(priceHistory, lookback = config.lstmLookbackPeriod) {
    const prices = priceHistory.map(d => d.price);
    const X = []; // Features (sequences)
    const y = []; // Labels (next price)

    for (let i = 0; i < prices.length - lookback; i++) {
        X.push(prices.slice(i, i + lookback));
        y.push(prices[i + lookback]);
    }

    if (X.length === 0) {
        return { X: tf.tensor([], [0, lookback, 1]), y: tf.tensor([], [0, 1]) };
    }

    const X_tensor = tf.tensor2d(X).reshape([-1, lookback, 1]);
    const y_tensor = tf.tensor2d(y, [y.length, 1]);

    return { X: X_tensor, y: y_tensor };
}


function prepareXgboostData(priceHistory) {
    const prices = priceHistory.map(d => d.price);
    const enrichedData = priceHistory.map(d => ({ ...d }));

    const rsiPeriod = config.indicatorPeriodRSI;
    if (prices.length >= rsiPeriod) {
        const rsiValues = TI.RSI.calculate({ values: prices, period: rsiPeriod });
        for (let i = 0; i < rsiValues.length; i++) {
            enrichedData[i + rsiPeriod - 1].rsi = rsiValues[i];
        }
    }

    if (prices.length >= config.macdSlowPeriod) {
        const macdResults = TI.MACD.calculate({
            values: prices,
            fastPeriod: config.macdFastPeriod,
            slowPeriod: config.macdSlowPeriod,
            signalPeriod: config.macdSignalPeriod
        });
        for (let i = 0; i < macdResults.length; i++) {
            enrichedData[i + config.macdSlowPeriod - 1].macd = macdResults[i].MACD;
            enrichedData[i + config.macdSlowPeriod - 1].macdSignal = macdResults[i].signal;
            enrichedData[i + config.macdSlowPeriod - 1].macdHistogram = macdResults[i].histogram;
        }
    }

    const minRequiredIndicatorLength = Math.max(rsiPeriod, config.macdSlowPeriod);
    const filteredData = enrichedData.slice(minRequiredIndicatorLength - 1).filter(d =>
        d.rsi !== undefined && d.macd !== undefined && d.macdSignal !== undefined
    );

    const features = [];
    const labels = [];

    for (let i = 0; i < filteredData.length - 1; i++) {
        const currentData = filteredData[i];
        const nextPrice = filteredData[i + 1].price;

        if (typeof currentData.rsi === 'number' && typeof currentData.macd === 'number' && typeof currentData.macdSignal === 'number') {
            features.push([
                currentData.rsi,
                currentData.macd,
                currentData.macdSignal
            ]);
            labels.push(nextPrice);
        }
    }

    return {
        enrichedData: filteredData,
        X: features,
        y: labels
    };
}

//function to build the model
function buildLstmModel(lookback) {
    const model = tf.sequential();
    model.add(tf.layers.lstm({ inputShape: [lookback, 1], units: 50, activation: 'relu', returnSequences: false }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({ optimizer: tf.train.adam(), loss: 'meanSquaredError' });
    return model;
}

//we traain the data. but call the function on retrain models

async function trainLstmModel(X_train, y_train, epochs = config.lstmTrainingEpochs) {
    if (!lstmModel) {
        lstmModel = buildLstmModel(X_train.shape[1]);
    }
    console.log('Starting LSTM model training...');
    await lstmModel.fit(X_train, y_train, { epochs: epochs, callbacks: {
        onEpochEnd: (epoch, logs) => {
            if (epoch % 10 === 0 || epoch === epochs - 1) {
                console.log(`LSTM Epoch ${epoch + 1}/${epochs}: Loss = ${logs.loss.toFixed(6)}`);
            }
        }
    }});
    console.log('LSTM model training complete.');
}

//make a prediction
function predictLstm(inputSequence) {
    if (!lstmModel) {
        console.warn('LSTM model not trained or loaded. Cannot make prediction.');
        return null;
    }
    const inputTensor = tf.tensor2d([inputSequence]).reshape([1, inputSequence.length, 1]);
    const prediction = lstmModel.predict(inputTensor);
    return prediction.dataSync()[0];
}

// === xGBoost Model functions ---

async function trainXgboostModel(X_train, y_train) {
    // XGBoost options can be tuned further
    const options = {
        objective: 'reg:squarederror', // Regression task
        nEstimators: 100, // Number of boosting rounds
        learningRate: 0.1,
        maxDepth: 3
    };

    console.log('Training XGBoost model...');
    xgboostModel = new XGBoost(options); // Create a new model instance
    await xgboostModel.train(X_train, y_train);
    console.log('XGBoost model trained.');
}


function predictXgboost(inputFeatures) {
    if(!xgboostModel){
        console.warn('XGBoost model not trained or loaded. Cannot make prediction.');
        return null;
    }

    // predict method expects an array of feature vectors
    const prediction = xgboostModel.predict([inputFeatures]);
    return prediction[0]; // get the single predicted value

}

// -- Model savig and  loadineg

export async function saveModels() {
    try {
        // create models directory if it doesnts exist
        const modelsDir = './models';
        await fs.mkdir(modelsDir, { recursive: true });

        if (lstmModel) {
            await lstmModel.save(LSTM_MODEL_PATH);
            console.log('LSTM model saved.');
        } else {
            console.warn('No LSTM model to save.');
        }

        //save the xgboost model
        if (xgboostModel) {
            const modelJson = xgboostModel.toJSON();
            await fs.writeFile(XGBOOST_MODEL_PATH, JSON.stringify(modelJson));
            console.log('XGBoost model saved.');
        } else {
            console.warn('No XGBoost model to save.');
        }
    }catch (error) {
        console.error('Error saving models:', error);
    }
}

export async function loadModels() {
    try {
        lstmModel = await tf.loadLayersModel(`${LSTM_MODEL_PATH}/model.json`);
        console.log('LSTM model loaded.');
    } catch (error) {
        console.warn('No saved LSTM model found or error loading:', error.message);
        lstmModel = null;
    }

    try {
        const modelJson = await fs.readFile(XGBOOST_MODEL_PATH, 'utf8');
        xgboostModel = await XGBoost.load(JSON.parse(modelJson));
        console.log('XGBoost model loaded.');
    } catch (error) {
        console.warn('No saved XGBoost model found or error loading:', error.message);
        xgboostModel = null;
    }
}


// --- Main Prediction Function ---

export async function generatePrediction(pairAddress) {
    console.log(`Generating prediction for ${pairAddress}...`);
    let lstmPrediction = null;
    let xgboostPrediction = null;

    try {
        const priceHistory = await dataService.getPriceHistory(pairAddress);

        // Ensure enough historical data for both models and indicators
        const minOverallData = Math.max(
            config.historyRetentionLimit,
            config.lstmLookbackPeriod + 1, // At least lookback + 1 for LSTM
            config.macdSlowPeriod + 1, // At least slow period + 1 for XGBoost
            config.indicatorPeriodRSI + 1
        );

        if (!priceHistory || priceHistory.length < minOverallData) {
            console.warn(`Not enough historical data for ${pairAddress} to generate full prediction. Need at least ${minOverallData} data points.`);
            return { combinedPrediction: null, lstmPrediction: null, xgboostPrediction: null, details: 'Not enough historical data for full prediction.' };
        }

        // --- LSTM Prediction ---
        const { X: lstmX, y: lstmY } = createLstmDataset(priceHistory);

        if (lstmX.shape[0] > 0) { // Check if dataset was successfully created
            await trainLstmModel(lstmX, lstmY);
            const latestPriceSequence = priceHistory
                .map(d => d.price)
                .slice(-config.lstmLookbackPeriod);

            if (latestPriceSequence.length === config.lstmLookbackPeriod) {
                lstmPrediction = predictLstm(latestPriceSequence);
                console.log(`LSTM Next Price Prediction: ${lstmPrediction.toFixed(8)}`);
            } else {
                console.warn(`Insufficient recent data for LSTM prediction for ${pairAddress}.`);
            }
        } else {
            console.warn(`Not enough samples for LSTM training or prediction for ${pairAddress}.`);
        }

        // --- XGBoost Prediction ---
        const { X: xgboostX, y: xgboostY, enrichedData } = prepareXgboostData(priceHistory);

        if (xgboostX.length > 0) { // Check if dataset was successfully created
            await trainXgboostModel(xgboostX, xgboostY);
            // Get the latest features for XGBoost prediction
            const lastEnrichedData = enrichedData[enrichedData.length - 1];
            if (lastEnrichedData && typeof lastEnrichedData.rsi === 'number' && typeof lastEnrichedData.macd === 'number' && typeof lastEnrichedData.macdSignal === 'number') {
                 const latestXgboostFeatures = [
                    lastEnrichedData.rsi,
                    lastEnrichedData.macd,
                    lastEnrichedData.macdSignal
                ];
                xgboostPrediction = predictXgboost(latestXgboostFeatures);
                console.log(`XGBoost Next Price Prediction: ${xgboostPrediction.toFixed(8)}`);
            } else {
                console.warn(`Insufficient recent features for XGBoost prediction for ${pairAddress}.`);
            }
        } else {
            console.warn(`Not enough samples for XGBoost training or prediction for ${pairAddress}.`);
        }

        // --- Combine Predictions ---
        let combinedPrediction = null;
        let details = '';

        if (lstmPrediction !== null && xgboostPrediction !== null) {
            // Simple average for demonstration; you can implement a more complex ensemble
            combinedPrediction = (lstmPrediction + xgboostPrediction) / 2;
            details = 'Combined prediction from LSTM and XGBoost.';
        } else if (lstmPrediction !== null) {
            combinedPrediction = lstmPrediction;
            details = 'Prediction from LSTM only (XGBoost data insufficient).';
        } else if (xgboostPrediction !== null) {
            combinedPrediction = xgboostPrediction;
            details = 'Prediction from XGBoost only (LSTM data insufficient).';
        } else {
            details = 'No valid predictions could be generated from either model.';
        }

        return {
            combinedPrediction: combinedPrediction,
            lstmPrediction: lstmPrediction,
            xgboostPrediction: xgboostPrediction,
            details: details
        };

    } catch (error) {
        console.error(`Error generating prediction for ${pairAddress}:`, error);
        return { combinedPrediction: null, lstmPrediction: null, xgboostPrediction: null, details: `Error generating prediction: ${error.message}` };
    }
}
// Add this to services/predictionService.js
export function hasModelsLoaded() {
    return lstmModel !== null && xgboostModel !== null;
}