
import math

def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    sqrt_n = math.isqrt(n)
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def nth_prime(n):
    primes_found = 0
    num_to_check = 2
    while True:
        if is_prime(num_to_check):
            primes_found += 1
            if primes_found == n:
                return num_to_check
        num_to_check += 1

def main():
    user_input = input("Enter the value of n: ")
    try:
        n = int(user_input)
        if n <= 0:
            print("Please enter a positive integer.")
        else:
            prime_number = nth_prime(n)
            print(f"The {n}th prime number is: {prime_number}")
    except ValueError:
        print("Invalid input. Please enter an integer.")

if __name__ == "__main__":
    main()
```