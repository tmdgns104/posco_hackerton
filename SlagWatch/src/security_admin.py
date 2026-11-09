"""Bootstrap the first local administrator from the server console."""
import argparse
import getpass

from .security import SecurityError, bootstrap_admin, create_user, login, logout


def main():
    parser = argparse.ArgumentParser(description="Create the first SteelTrust local administrator.")
    parser.add_argument("command", choices=["bootstrap", "create-user"])
    parser.add_argument("username")
    parser.add_argument("--role", choices=["admin", "operator", "viewer"], default="viewer")
    args = parser.parse_args()
    password = getpass.getpass("New account password (at least 12 characters): ")
    confirmation = getpass.getpass("Repeat password: ")
    if password != confirmation:
        raise SecurityError("Passwords do not match.")
    if args.command == "bootstrap":
        bootstrap_admin(args.username, password)
        print("Administrator created. The temporary local audit store is used unless an external destination is configured.")
    else:
        admin = input("Administrator account: ")
        token = login(admin, getpass.getpass("Administrator password: "))
        try:
            create_user(token, args.username, password, args.role)
        finally:
            logout(token)
        print("Account created.")


if __name__ == "__main__":
    main()
