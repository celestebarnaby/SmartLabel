from benchmark import Benchmark


# NEW BENCHMARKS
mnist_benchmarks = [
        # Robert
        Benchmark(
            "(length (filter (curry le 5) (map (curry plus 1) (map_imgs pred_int input-list))))",
            "Counts how many exam scores are still low (below 5) after adding an extra credit point."
        ),
        Benchmark(
            "(fold max 0  (map (curry plus (apply pred_int input-img)) (filter (curry le 9) (map_imgs pred_int input-list))))",
            "Calculates the highest score after x extra credit points are added to scores below 9."
        ),
        Benchmark(
            "(fold plus 0 (map (curry plus (apply pred_int input-img)) (filter (curry le 9) (map_imgs pred_int input-list))))",
            "For cheap products being sold at a store, calculates the total revenue from selling out these products after increasing their price by x dollars.",
        ),
        Benchmark(
            "(length (filter (curry ge (apply pred_int input-img)) (map (curry mult 2)  (map_imgs pred_int input-list))))",
            "After doubling the amount every product in the inventory, count how many have more than x units in stock, where x is the required minimum."
        ),
        Benchmark(
            "(fold max 0 (map (curry plus (apply pred_int input-img)) (filter (curry le 2)  (map_imgs pred_int input-list))))",
            "Researchers at the North Pole took notes of readings from a temperature sensor that is inaccurate in extreme cold. For temperatures below 2 Fahrenheit, corrects the false temperature readings by increasing them by x. Then reports the highest temperature at which the sensor starts failing."
        ),
        # Shankara
        Benchmark(
            "(fold plus 0 (filter (curry ge (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "Sum of digits greater than x"
        ),
        Benchmark(
            "(fold mult 1 (filter (curry le (apply pred_int input-img)) (map (curry mult 3) (map_imgs pred_int input-list))))",
            "Product of digits less than x after multiplying by 3"
        ),
        # # TODO: this doesn't parse and also the description is wrong?
        # # 3. Numerator of the Second Central Moment
        # # 𝜆 𝑙, 𝑥 . fold sum 0 (map (curry product (curry sum 𝑥)) 𝑙)

        # # TODO: not supported by our DSL
        Benchmark(
            "(length (filter (curry le 8) (filter (curry ge 8) (map_imgs pred_int input-list))))",
            "Computing the number of occurences of a single digit"
        ),

        # # TODO: not supported by DSL
        # # 5. Computing Maximum and Minimum Values in List
        # # 𝜆 𝑙, 𝑥 . (fold max 0 𝑙, fold min 9 𝑙)

        # # Anirudh

        Benchmark(
        "(fold plus (apply pred_int input-img) (filter (curry ge (apply pred_int input-img)) (map_imgs pred_int input-list)))",
        "Calculate the total amount of donations exceeding a specific threshold, plus that threshold added"
        ),

        Benchmark(
        "(fold plus 0 (map (curry mult 2) (map_imgs pred_int input-list)))",
        "Calculate the total amount of donations that have been matched"
        ),

        Benchmark(
        "(fold max (apply pred_int input-img) (filter (curry ge 9) (map_imgs pred_int input-list)))",
        "Determine the maximum age among a group of at least age 10"
        ),
        Benchmark(
        "(fold plus 0 (map (curry plus (apply pred_int input-img)) (map_imgs pred_int input-list)))",
        "Adjust the inventory count after restocking"
        ),
        Benchmark(
        "(length (filter (curry le (apply pred_int input-img)) (map_imgs pred_int input-list)))",
        "Count the number of small retail transactions below a limit"
        ),
        Benchmark(
        "(fold mult 1 (map (curry mult 7) (map_imgs pred_int input-list)))",
        "Compute the product of numbers after multiplying by a specific number"
        ),

        # Noah?        
        Benchmark(
            "(length (filter (curry le 9) (filter (curry ge 5) (map_imgs pred_int input-list))))",
            "Counts the number of participants in a study between age 5 and 9"
        ),
        Benchmark(
            "(fold max 0 (filter (curry le 2) (map_imgs pred_int input-list)))",
            "Finds the maximum sub 2k dollar expenditure on a balance sheet."
        ),
        Benchmark(
            "(fold plus 0 (filter (curry ge 8) (map_imgs pred_int input-list)))",
            "Finds the total expenditures from expenditures over 8k dollars on a balance sheet."
        ),
        Benchmark(
            "(fold max 0 (map (curry mult 2) (map_imgs pred_int input-list)))",
            "Finds the size of the largest class if every class size were to double."
        ),
        Benchmark(
            "(fold max 0 (map (curry mult 2) (filter (curry ge 5) (map_imgs pred_int input-list))))",
            "Finds the size of the largest class, if every class with less than 5 students dissolved and every other class size doubled."
        ),


        # Zetten

        Benchmark(
            "(length (filter (curry ge 1) (map_imgs pred_int input-list)))",
            "Number of positive, non-zero numbers in the list"
        ),
        Benchmark(
            "(fold mult 1 (map_imgs pred_int input-list))",
            "Cumulative compound interest by years"
        ),

    # TODO: don't understand this program?
    # Explanation: This program determines the maximum difference in product prices from a given list of prices. It calculates two sequential maximums to get the maximum difference between them using a curried function with a parameter x

    # \lambda l, x. fold max 0 (map (curry sum x) (filter (\y .y > x) (map (curry max 0) l)))
        Benchmark(
            "(fold max 0 (map_imgs pred_int input-list))",
            "Calculating maximum score on a midterm"
        ), 
        Benchmark(
            "(length (filter (curry ge (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "Calculating number of people with a passing grade"
        ), 

    # Ruijie

# TODO: not supported by DSL
# \l1.\l2. fold(sum, 0, map((\x.map(\y.product(x,y),l2), l1)) - given lists [c1,...,cn] and [d1,...,dn] as coefficients to two n-degree polynomials, construct the list [c1*d1,...,c1*dn,...,cn*d1,...,cn*dn] of coefficients for the product of the two polynomials, and then sum everything up.

        Benchmark(
            "(length (filter (curry le 0) (map_imgs pred_int input-list)))",
            "Count the number of zeroes in l"
        ), 

# TODO: not supported by DSL
# \l1. \l2. fold(sum, 0, filter(x>0, map(\x.map(\y.sum(x,y),l2),l1))) - given lists [a1,...,an] and [b1,...,bn], take pairwise sum of form a_i+b_j and filter out the non-positive elements, and sum everything up
# in resultant list.

        Benchmark(
            "(fold max 0 (filter (curry ge 5) (map_imgs pred_int input-list)))",
            "Take the max among the elements >5 of the list."
        ), 

# TODO: not supported by DSL
# \l0. l1. max(fold(\s.\x.max(s,x), 0, filter(\x.x>0,l0)), fold(\s.\x.max(s,x), 0, filter(\x.x>0,l1))) - compute the max among positive elements of list l0, and then list l1, and take the max of those two.

    # Sadanand

        Benchmark(
            "(fold max 0 (filter (curry le (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "Determine the highest score among all scores that are below a given threshold x. This can be used, for example, to identify the highest failing grade in a class, helping educators understand which students are closest to passing and may benefit from additional support."
        ),

        Benchmark(
            "(fold plus 0 (map (curry mult 6) (map_imgs pred_int input-list)))",
            "Compute the total sales revenue by multiplying the quantity of each item sold (given in list l) by the unit price 6 and summing the results. This is useful for businesses to calculate total income from sales when provided with quantities of items sold and a constant unit price."
        ),
        Benchmark(
            "(fold plus 0 (filter (curry ge 2) (map_imgs pred_int input-list)))",
            "Calculate the total number of batteries in boxes containing at least 2 batteries, based on the handwritten number written on the boxes as detected by the robot's camera. This helps the robot focus on larger battery packs during inventory management."
        ),
        Benchmark(
            "(length (filter (curry ge (apply pred_int input-img)) (map (curry plus 4) (map_imgs pred_int input-list))))",
            "Determine the number of students who would pass the course after adding 4 bonus points to each of their grades, given a passing mark x. This helps educators assess the impact of a grading curve or extra credit on overall student success rates."
        ),
        Benchmark(
            "(length (filter (curry ge (apply pred_int input-img)) (filter (curry le 7) (map_imgs pred_int input-list))))",
            "Determine the number of objects within a size range suitable for the robot's gripper, by filtering objects larger than a minimum size x and less than a maximum size (e.g., 5 units). This helps the robot identify which objects it can manipulate effectively without exceeding its mechanical limits."
        ),

    # Maxine
        Benchmark(
            "(length (filter (curry ge 1) (map_imgs pred_int input-list)))",
            "This program counts the number of conference attendees who have more than 0 dietary restrictions."
        ), 
        Benchmark(
            "(fold plus 0 (map (curry mult (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "This program calculates the total weekly egg budget. It assumes that each person needs a specific number of eggs per day, reflected on the list l, and each egg costs x dollars."
        ),
        Benchmark(
            "(fold max 0 (map (curry plus (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "This program identifies the most expensive neutral oil in the grocery store, adding x dollars of tax."
        ),
        Benchmark(
            "(length (filter (curry ge 4) (map_imgs pred_int input-list)))",
            "This program counts the number of restaurants with ratings greater than 4 from the list of restaurant ratings."
        ), 
        Benchmark(
            "(fold max 0 (map (curry plus 5) (map_imgs pred_int input-list)))",
            "This program identifies the microwave recipe with the longest cooking time, adding 5 minute for prep."
        ),

    # Tony 

        Benchmark(
            "(length (filter (curry ge 9) (map_imgs pred_int input-list)))",
            "Count the total expenses of large financial transactions that exceeds 9k dollars."
        ), 
        Benchmark(
            "(length (filter (curry ge 3) (map_imgs pred_int input-list)))",
            "In the pool of application of phd at University of Texas at Austin, count how many applicants got a gpa of at least 3."
        ), 
        Benchmark(
            "(length (filter (curry le (apply pred_int input-img)) (map (curry plus 2) (map_imgs pred_int input-list))))",
            "In automated logical reasoning class, count how many students still can not get C grade where the cutoff is x event after a bonus point of 2."
        ), 
        Benchmark(
            "(length (filter (curry le (apply pred_int input-img)) (map (curry plus 1) (map_imgs pred_int input-list))))",
            "Count how many reviewers from SOSP have ratings below x, if each rating were increased by 1"
        ), 
        Benchmark(
            "(fold plus 1 (map (curry max 8) (filter (curry ge 4) (map_imgs pred_int input-list))))",
            "In a jewelry store, calculate the total price when all the products below or equal to 4 are removed from shelves while the remaining products are lifted to at least 8 dollars to counter inflation"
        ),


    # Anon

        Benchmark(
            "(fold plus 0 (filter (curry le (apply pred_int input-img)) (map_imgs pred_int input-list)))",
            "Find the total score among all football teams scoring less than x points."
        ),
        Benchmark(
            "(fold mult 1 (map (curry max 1) (map_imgs pred_int input-list)))",
            "Find the product of all numbers in a list, where 0 is replaced by 1."
        ),
        Benchmark(
            "(fold mult 1 (filter (curry ge 2 ) (filter (curry le 4) (map_imgs pred_int input-list))))",
            "Find the product of all numbers in a list between 1 and 5 exclusive."
        ),
        Benchmark(
            "(fold plus 0 (map (curry mult 3) (map_imgs pred_int input-list)))",
            "Find the total score in the last round of Family Feud, where points are worth triple."
        ),
        Benchmark(
            "(length (filter (curry ge 1) (filter (curry le 9) (map_imgs pred_int input-list))))",
            "Find the number of single-digit, non-zero race times."
        ), 

        # Linus
        Benchmark(
            "(fold plus 0 (filter (curry ge 3) (map_imgs pred_int input-list)))",
            "Given a bunch of purchases at a cash register, get the sum of the transactions over $2."
        ),
        Benchmark(
            "(length (filter (curry ge 3) (map (curry plus 1) (map_imgs pred_int input-list))))",
            "In the pool of application of phd at University of Texas at Austin, count how many applicants got a gpa of at least 3."
        ), 
        Benchmark(
            "(fold plus 0 (map (curry plus 2) (map (curry mult (apply pred_int input-img)) (map_imgs pred_int input-list))))",
            "Given a list of how many items are in each box, where the cost per item is x, and the flat cost of a box is $2, how much does everything cost?"
        ),
        Benchmark(
            "(fold max 0 (filter (curry le 9) (map_imgs pred_int input-list)))",
            "We are trying to get rid of 8kg of rice by giving it to someone. Given a list of requests of people wanting various amounts of rice, find the one which requests the most"
        ),
        Benchmark(
            "(fold max 0 (filter (curry le (apply pred_int input-img)) (map (curry mult 5) (map_imgs pred_int input-list))))",
            "Look for the max weight of an item so that you can put 5 of them on a lift that has a max weight of x."
        )
]

# # OLD BENCHMARKS
old_mnist_benchmarks = [
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