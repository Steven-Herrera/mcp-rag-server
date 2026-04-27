# mcp-rag-server
A containerized Retrieval Augmented Generation (RAG) Model Context Protocol (MCP) Server using Qdrant as the database. 

## Environment
This RAG MCP server is deployed to a High Availability (HA) 3-node proxmox cluster running 3 Virtual Machines (VMs) which in turn are hosting a 3-node HA k3s cluster. The k3s cluster reside in a Virtual Local Area Network (VLAN) that is separate from the proxmox cluster's LAN.

The 3 nodes are:
* a 2012 iMac
* Dell Optiplex
* ASUS Laptop

Put simply, the nodes are repurposed machines given a new life using proxmox. 

## RAG Database
Qdrant is used for as the backend for the RAG database. 
