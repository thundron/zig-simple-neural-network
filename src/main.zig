const std = @import("std");
const rand = std.rand;

const INPUT_NODES = 2;
const HIDDEN_NODES = 4;
const OUTPUT_NODES = 1;
const TRAINING_EPOCHS = 10000;
const LEARNING_RATE = 0.1;

// Structure to hold weights
const Weights = struct {
    hidden: [][]f64,
    output: [][]f64,
    hiddenBias: []f64,
    outputBias: []f64,
};

// Sigmoid function
fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

// Derivative of sigmoid
fn sigmoidDerivative(x: f64) f64 {
    return x * (1 - x);
}

// Initialize weights
fn initializeWeights(allocator: *const std.mem.Allocator, inputNodes: u32, hiddenNodes: u32, outputNodes: u32) !Weights {
    var weights = Weights{
        .hidden = try allocator.alloc([]f64, hiddenNodes),
        .output = try allocator.alloc([]f64, outputNodes),
        .hiddenBias = try allocator.alloc(f64, hiddenNodes),
        .outputBias = try allocator.alloc(f64, outputNodes),
    };

    var prng = rand.DefaultPrng.init(0);
    const random = prng.random();

    for (weights.hidden, 0..) |*hidden, i| {
        hidden.* = try allocator.alloc(f64, inputNodes);
        for (hidden.*) |*weight| {
            weight.* = random.float(f64) * 2 - 1;
        }
        weights.hiddenBias[i] = random.float(f64) * 2 - 1;
    }

    for (weights.output, 0..) |*output, i| {
        output.* = try allocator.alloc(f64, hiddenNodes);
        for (output.*) |*weight| {
            weight.* = random.float(f64) * 2 - 1;
        }
        weights.outputBias[i] = random.float(f64) * 2 - 1;
    }

    return weights;
}

// Forward propagation
fn forward(weights: *Weights, inputs: *const [2]f64, output: []f64, hiddenLayer: []f64) void {
    for (hiddenLayer, 0..) |*hidden, i| {
        var sum: f64 = weights.hiddenBias[i];
        for (inputs, 0..) |input, j| {
            sum += input * weights.hidden[i][j];
        }
        hidden.* = sigmoid(sum);
    }

    for (output, 0..) |*out, i| {
        var sum: f64 = weights.outputBias[i];
        for (hiddenLayer, 0..) |hidden, j| {
            sum += hidden * weights.output[i][j];
        }
        out.* = sigmoid(sum);
    }
}

// Train the network
fn train(weights: *Weights, inputs: *const [2]f64, target: f64, learningRate: f64) void {
    var output: [OUTPUT_NODES]f64 = undefined;
    var hiddenLayer: [HIDDEN_NODES]f64 = undefined;
    forward(weights, inputs, &output, &hiddenLayer);

    const err = target - output[0];
    const outputDelta = err * sigmoidDerivative(output[0]);

    // Update output weights and biases
    for (weights.output, 0..) |*outputWeights, i| {
        for (outputWeights.*, 0..) |*weight, j| {
            weight.* += learningRate * outputDelta * hiddenLayer[j];
        }
        weights.outputBias[i] += learningRate * outputDelta;
    }

    // Backpropagate error to hidden layer
    var hiddenDeltas: [HIDDEN_NODES]f64 = undefined;
    for (&hiddenDeltas, 0..) |*delta, j| {
        delta.* = outputDelta * weights.output[0][j] * sigmoidDerivative(hiddenLayer[j]);
    }

    // Update hidden weights and biases
    for (weights.hidden, 0..) |*hiddenWeights, i| {
        for (hiddenWeights.*, 0..) |*weight, j| {
            weight.* += learningRate * hiddenDeltas[i] * inputs[j];
        }
        weights.hiddenBias[i] += learningRate * hiddenDeltas[i];
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var weights = try initializeWeights(&allocator, INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES);
    defer {
        for (weights.hidden) |hidden| allocator.free(hidden);
        for (weights.output) |output| allocator.free(output);
        allocator.free(weights.hidden);
        allocator.free(weights.output);
        allocator.free(weights.hiddenBias);
        allocator.free(weights.outputBias);
    }

    // XOR training data
    const trainingData = [_][INPUT_NODES + 1]f64{
        .{ 0, 0, 0 },
        .{ 0, 1, 1 },
        .{ 1, 0, 1 },
        .{ 1, 1, 0 },
    };

    // Train the network
    var epoch: u32 = 0;
    while (epoch < TRAINING_EPOCHS) : (epoch += 1) {
        for (trainingData) |data| {
            const inputs = data[0..INPUT_NODES];
            train(&weights, inputs, data[2], LEARNING_RATE);
        }
    }

    // Test the network
    for (trainingData) |data| {
        const inputs = data[0..INPUT_NODES];
        var output: [OUTPUT_NODES]f64 = undefined;
        var hiddenLayer: [HIDDEN_NODES]f64 = undefined;
        forward(&weights, inputs, &output, &hiddenLayer);
        std.debug.print("Input: {:.2}, {:.2}, Predicted: {:.2}, Target: {:.2}\n", .{ inputs[0], inputs[1], output[0], data[2] });
    }
}
