---
name: qdrant-docker
description: Checks that the Qdrant server is up on localhost:6333 and, if not, starts Docker Desktop and the Qdrant container. Use before running scripts/pipeline.sh or anything that builds or queries the l4 Qdrant index (l4 indexing always builds a Qdrant collection on this server, so the pipeline fails without it), or when the user asks to check, start, or run Qdrant / Docker.
---

# Qdrant in Docker

Do all steps in order without asking permission for each one. Stop and report
to the user at any step that says so. Never remove, recreate or prune a
container, image or volume.

The l4 indexing layer connects to this server at `QDRANT_URL`
(`http://localhost:6333`, in `src/common/constants.py`) and stores one
collection per chunking run, named by
`common.paths.get_qdrant_collection_name`. Qdrant data is bind-mounted at
`/home/tanzima/Documents/AI/qdrant`, so collections survive container
restarts and act as the Qdrant index cache. Never delete a collection unless
the user asks.

If `QDRANT_URL` in `constants.py` no longer points at `localhost:6333`, use
its host and port in the checks below instead.

## 1. Check if Qdrant is already up

```bash
curl -sf -m 3 http://localhost:6333 >/dev/null && echo UP || echo DOWN
```

If `UP`, skip to step 5.

## 2. Start Docker Desktop if it isn't running

Docker Desktop runs as a user-level systemd service, so no `sudo` is needed.
Starting it when it already runs does nothing.

```bash
systemctl --user start docker-desktop
```

Wait for the daemon, up to about 2 minutes:

```bash
for i in $(seq 1 24); do
  docker ps >/dev/null 2>&1 && echo DOCKER_UP && break
  sleep 5
done
```

If `DOCKER_UP` is not printed, stop and report it with the output of
`systemctl --user status docker-desktop --no-pager | tail -20`.

## 3. Start the Qdrant container

List existing Qdrant containers, newest first:

```bash
docker ps -a --filter ancestor=qdrant/qdrant --format '{{.Names}}\t{{.Status}}\t{{.CreatedAt}}'
```

- **One is `Up`**: go to step 4.
- **Only stopped ones**: start the newest one (top of the list). Leave older
  ones alone, they may be backups.
  ```bash
  docker start <container_name>
  ```
- **None**: create one. The bind mount keeps the data on disk.
  ```bash
  docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
    -v /home/tanzima/Documents/AI/qdrant:/qdrant/storage qdrant/qdrant
  ```

If `docker start` or `docker run` fails because port 6333 is in use, stop and
report which process or container holds it (`docker ps --filter publish=6333`).

## 4. Confirm it's reachable

```bash
for i in $(seq 1 10); do
  curl -sf -m 3 http://localhost:6333 >/dev/null && echo QDRANT_UP && break
  sleep 2
done
curl -s http://localhost:6333
```

Expect `{"title":"qdrant - vector search engine", ...}`. If `QDRANT_UP` is not
printed, stop and show `docker logs --tail 30 <container_name>`.

## 5. Report

Tell the user in one or two lines: whether Qdrant was already up or what was
started, the container name, and the Qdrant version from the response. Then
continue with what the user asked for, such as running `scripts/pipeline.sh`.
