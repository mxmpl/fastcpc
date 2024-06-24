.PHONY: clean linter

clean:
	rm -f *.out *.err

linter:
	ruff format src
	ruff check src
	typos
