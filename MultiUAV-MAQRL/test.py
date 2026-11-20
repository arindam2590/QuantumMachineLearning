"""General deterministic checkpoint testing and rendering entry point."""
from algorithms.runtime import evaluation_main

def main(argv=None):
    return evaluation_main("testing",argv)

if __name__ == "__main__":
    main()
