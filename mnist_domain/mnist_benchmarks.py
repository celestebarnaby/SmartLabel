from benchmark import Benchmark

mnist_benchmarks = [
        Benchmark(
            "(fold plus 0 (map_imgs pred_int input-list))",
            "sum of elements",
        ),
        Benchmark(
            "(fold plus (apply pred_int input-img) (map_imgs pred_int input-list))",
            "sum of elements and k",
        ),
        Benchmark(
            "(fold plus 0 (map (curry plus 1) (map_imgs pred_int input-list)))",
            "add 1 to all elements and take sum",
        ),
        Benchmark(
            "(fold plus 0 (map (curry plus (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "add k to all elements and take sum",
        ),
        Benchmark(
            "(fold plus 0 (map (curry mult 2) (map_imgs pred_int input-list)))",
            "multiply all elements by 2 and take sum",
        ),
        Benchmark(
            "(fold plus 0 (map (curry mult 2) (map (curry plus 1) (map_imgs pred_int input-list))))",
            "add 1 to all elements, multiply by 2, and take sum",
        ),
        Benchmark(
            "(fold plus 0 (map (curry mult (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "multiply all elements by k and take sum",
        ),
        Benchmark(
            "(fold mult 1 (map_imgs pred_int input-list))",
            "return product",
        ),
        Benchmark(
            "(fold mult (apply pred_int input-img) (map_imgs pred_int input-list))",
            "return product of all elements and k",
        ),
        Benchmark(
            "(fold mult 1 (map (curry plus 1) (map_imgs pred_int input-list)))",
            "add 1 to all elements and take product",
        ),
    
        Benchmark(
            "(fold mult 1 (map (curry plus (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "add k to all elements and take product",
        ),
        Benchmark(
            "(fold max (apply pred_int input-img) (map_imgs pred_int input-list))",
            "max of elements compared with k",
        ),
        Benchmark(
            "(fold max (apply pred_int input-img) (map (curry mult 2) (map_imgs pred_int input-list)))",
            "multiply elements by 2 and take max compared with k",
        ),
        Benchmark(
            "(fold max (apply pred_int input-img) (map (curry plus 2) (map_imgs pred_int input-list)))",
            "add 2 to all elements and take max compared with k",
        ),
        Benchmark(
            "(fold plus 0 (filter (curry ge (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "add elements greater than k",
        ),
        Benchmark(
            "(fold mult 1 (filter (curry ge (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "multiply elements greater than k",
        ),
        Benchmark(
            "(fold plus 0 (filter (curry le (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "add elements less than k",
        ),
        Benchmark(
            "(fold mult 1 (filter (curry le (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "multiply elements less than k",
        ),
        Benchmark(
            "(fold mult 1 (filter (curry le (apply pred_int input-img)) (map (curry mult 2) (map_imgs pred_int input-list))))",
            "multiply elements by 2, filter elements less than k, and return product",
        ),

        Benchmark(
            "(length (filter (curry ge (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "number of elements greater than k",
        ),
        Benchmark(
            "(length (filter (curry le (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "number of elements less than k",
        ),
        Benchmark(
            "(length (filter (curry le 9) (map_imgs pred_int input-list)))",
            "number of elements less than 9",
        ),
        Benchmark(
            "(length (filter (curry le (apply pred_int input-img)) (map (curry mult 2) (map_imgs pred_int input-list))))",
            "multiply elements by 2, filter elements less than k, take length of list",
        ),
        Benchmark(
            "(length (filter (curry le (apply pred_int input-img)) (filter (curry ge 9) (map_imgs pred_int input-list))))",
            "number of elements greater than 9 and less than k",
        ),
        Benchmark(
            "(fold mult (apply pred_int input-img) (filter (curry le 9) (map_imgs pred_int input-list)))",
            "multiply elements less than 9",
        )
    ]