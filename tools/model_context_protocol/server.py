from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "MPC-count-total-rs",
    host="127.0.0.1",
    port=8080,
    timeout=30,
)


@mcp.tool()
def count_total_rs(text: str) -> int:
    """Count the total number of Rs in the given string.

    Input:
        text: str -> text to count the total number of Rs in

    Output:
        count: int -> total number of Rs in the given string
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    return text.lower().count("r")


if __name__ == "__main__":
    print("Starting MCP server...")
    mcp.run()
