import os
import uvicorn


def main() -> None:
    raw_port = (os.environ.get("PORT") or "8000").strip()
    try:
        port = int(raw_port)
    except ValueError:
        port = 8000

    host = (os.environ.get("HOST") or "0.0.0.0").strip() or "0.0.0.0"
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
