import uvicorn

from openenv_compliance_audit.server import app  # noqa: F401


def main() -> None:
    uvicorn.run("openenv_compliance_audit.server:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
