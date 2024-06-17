const std = @import("std");

pub fn main() !void {
    const input_layers = [3][4]f64{ [4]f64{
        0,
        0,
        1,
        1,
    }, [4]f64{
        1,
        1,
        1,
        0,
    }, [4]f64{
        1,
        0,
        1,
        1,
    } };

    const output_layer = [4]f64{ 0, 1, 1, 0 };
    var weights: [3]f64 = undefined;

    const rand = std.crypto.random;
    for (&weights) |*w| {
        w.* = 2 * rand.float(f64) - 1;
    }

    var iteration: u32 = 0;
    while (iteration < 10_000) : (iteration += 1) {
        var output: f64 = 0;
        for (input_layers) |layer| {
            for (layer) |v| {
                output += v * weights[iteration % 3];
            }
        }

        // apply sigmoid fn
        output = 1 / (1 + std.math.exp(-output));
        for (&weights, 0..) |*w, i| {
            for (input_layers[i]) |v| {
                w.* += v * (output_layer[0] - output) * output * (1 - output);
            }
        }
    }

    const input: [3]f64 = .{ 1, 0, 0 };
    var result: f64 = 0;
    for (input, 0..) |x, i| {
        result += x * weights[i];
    }
    result = 1 / (1 + std.math.exp(-result));

    std.debug.print("{d}\n", .{result});
}
