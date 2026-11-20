"""General deterministic headless validation entry point."""
from algorithms.runtime import evaluation_main

def main(argv=None):
    return evaluation_main("validation",argv)

if __name__ == "__main__":
    main()
