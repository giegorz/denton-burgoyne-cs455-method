from dataclasses import dataclass

@dataclass
class Point:
    id: int
    x: float
    y: float

    def to_dict(self):
        return {"id": self.id, "x": self.x, "y": self.y}

@dataclass
class Element:
    id: int
    points: list[Point]

@dataclass
class PointCollection:
    points: list[Point]

    def to_dict(self):
        return {"points": [p.to_dict() for p in self.points]}

@dataclass
class Value:
    point: Point
    gamma: float
    theta: float

    def to_array(self):
        return [
            self.point.x,
            self.point.y,
            self.gamma,
            self.theta
        ]