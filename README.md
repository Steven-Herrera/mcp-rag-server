# mcp-rag-server
A containerized Retrieval Augmented Generation (RAG) Model Context Protocol (MCP) Server using Qdrant as the database. This RAG MCP server is deployed to a High Availability (HA) 3-node proxmox cluster hosting a 3-node HA k3s cluster. The k3s cluster reside in a Virtual Local Area Network (VLAN) that is separate from the proxmox cluster's LAN.
