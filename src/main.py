from orchestrator import run

if __name__ == "__main__":
    response = run("Найди баги в коде и напиши документацию")
    print("FINAL ANSWER:")  # noqa: T201
    print(response.choices[0].message.content)  # noqa: T201
