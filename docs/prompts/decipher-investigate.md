Investigate a cipher with the Decipher MCP server.

If the `decipher` MCP server is not connected: run `sh scripts/bootstrap.sh`,
then reconnect (trust the project if prompted) and retry.

If I did not paste a cipher yet, ask me to paste it (or name a local text
file for you to read and pass inline — the server takes no file paths).
Then: call `investigation_start` with the inline ciphertext; read
`docs/mcp_onboarding.md` §Investigation methodology; and drive the
investigation from `investigation_status`, following its advisory guidance
until a verified declaration or an honest unsolved.
