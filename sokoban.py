import pygame
import sys

# ---------- Settings ----------
TILE = 48
FPS = 60

# Colors
BG = (30, 30, 30)
WALL = (70, 70, 70)
FLOOR = (50, 50, 50)
GOAL = (200, 170, 0)
BOX = (160, 82, 45)
BOX_ON_GOAL = (0, 200, 0)
PLAYER = (70, 130, 255)

DIRS = {
    pygame.K_w: (-1, 0),
    pygame.K_s: (1, 0),
    pygame.K_a: (0, -1),
    pygame.K_d: (0, 1),
}

LEVELS = [
    [
        "  #####  ",
        "###   #  ",
        "#.@$  ###",
        "### $.  #",
        "#.##    #",
        "#   #####",
        "#####    ",
    ],
    [
        "  ####### ",
        "###     # ",
        "#. $ $  # ",
        "#   ## ###",
        "#  @     #",
        "#######  #",
        "      ####",
    ],
]

# ---------- Game Logic ----------

def parse_level(level):
    walls, goals, boxes = set(), set(), set()
    player = None
    rows = len(level)
    cols = max(len(row) for row in level)

    for r in range(rows):
        for c in range(cols):
            ch = level[r][c] if c < len(level[r]) else " "
            if ch == "#":
                walls.add((r, c))
            elif ch == ".":
                goals.add((r, c))
            elif ch == "@":
                player = (r, c)
            elif ch == "+":
                player = (r, c)
                goals.add((r, c))
            elif ch == "$":
                boxes.add((r, c))
            elif ch == "*":
                boxes.add((r, c))
                goals.add((r, c))

    return {
        "walls": walls,
        "goals": goals,
        "boxes": boxes,
        "player": player,
        "rows": rows,
        "cols": cols,
    }

def try_move(state, dr, dc):
    pr, pc = state["player"]
    nr, nc = pr + dr, pc + dc

    if (nr, nc) in state["walls"]:
        return

    if (nr, nc) in state["boxes"]:
        br, bc = nr + dr, nc + dc
        if (br, bc) in state["walls"] or (br, bc) in state["boxes"]:
            return
        state["boxes"].remove((nr, nc))
        state["boxes"].add((br, bc))
        state["player"] = (nr, nc)
    else:
        state["player"] = (nr, nc)

def is_win(state):
    return state["boxes"] == state["goals"]

# ---------- Drawing ----------

def draw(screen, state):
    screen.fill(BG)

    for r in range(state["rows"]):
        for c in range(state["cols"]):
            rect = pygame.Rect(c*TILE, r*TILE, TILE, TILE)
            pygame.draw.rect(screen, FLOOR, rect)

            if (r, c) in state["walls"]:
                pygame.draw.rect(screen, WALL, rect)

            if (r, c) in state["goals"]:
                pygame.draw.circle(screen, GOAL, rect.center, TILE//6)

            if (r, c) in state["boxes"]:
                color = BOX_ON_GOAL if (r, c) in state["goals"] else BOX
                pygame.draw.rect(screen, color, rect.inflate(-10, -10))

    pr, pc = state["player"]
    rect = pygame.Rect(pc*TILE, pr*TILE, TILE, TILE)
    pygame.draw.circle(screen, PLAYER, rect.center, TILE//3)

# ---------- Main ----------

def main():
    pygame.init()

    level_index = 0

    def load_level(i):
        lvl = parse_level(LEVELS[i])
        return lvl

    state = load_level(level_index)

    width = state["cols"] * TILE
    height = state["rows"] * TILE

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Sokoban - WASD")

    clock = pygame.time.Clock()

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key in DIRS:
                    dr, dc = DIRS[event.key]
                    try_move(state, dr, dc)

                if event.key == pygame.K_r:
                    state = load_level(level_index)

                if event.key == pygame.K_n:
                    level_index = (level_index + 1) % len(LEVELS)
                    state = load_level(level_index)

                if event.key == pygame.K_p:
                    level_index = (level_index - 1) % len(LEVELS)
                    state = load_level(level_index)

                if event.key == pygame.K_ESCAPE:
                    running = False

        if is_win(state):
            pygame.display.set_caption("Sokoban - Level Complete! Press N")
        else:
            pygame.display.set_caption("Sokoban - WASD")

        draw(screen, state)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
