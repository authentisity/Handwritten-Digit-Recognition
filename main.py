'''
author: Hankang Li
'''

import tensorflow as tf
import numpy, pygame, math
from PIL import Image




model = tf.keras.models.load_model("model.h5")
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

pygame.init()
pygame.display.get_window_size
WIN_H, WIN_W = 1000, 1000
screen = pygame.display.set_mode((WIN_W, WIN_H))
screen.fill((255, 255, 255))


class button():

    instances = []
    def __init__(self, pos, size, color = (255, 0, 0), txt = "Title"):
        self.pos, self.size, self.color, self.txt = pos, size, color, txt
        self.rect = pygame.draw.rect(screen, self.color, (self.pos, self.size))
        screen.blit(pygame.font.Font(pygame.font.get_default_font(), int(self.size[0] / len(self.txt) * 2)).render((self.txt), True, (0, 0, 0)), self.rect.move(0, self.size[1] / 2 - self.size[0] / len(self.txt)))
        button.instances.append(self)

    @classmethod
    def get_pressed(cls, pos):
        for i in cls.instances:
            if i.rect.contains(pos, (0, 0)):
                return i.txt
        return None

class pixel():

    instances, ind = [], 0
    highlighted_boxes = []

    def __init__(self):
        self.pos = (pixel.pos[0] + pixel.size * (pixel.ind % 28), pixel.pos[1] + pixel.size * int(pixel.ind / 28))
        self.ind = pixel.ind
        pixel.ind += 1
        self.color = (0, 0, 0)
        pygame.draw.rect(screen, self.color if self.ind not in pixel.highlighted_boxes else (255, 255, 255), (self.pos[0], self.pos[1], self.size, self.size))
        pixel.instances.append(self)

    def update(self, color):
        self.color = tuple([min(self.color[0] + color, 255) for i in range(3)])
        pygame.draw.rect(screen, self.color, (self.pos[0], self.pos[1], self.size, self.size))


    @classmethod
    def init(cls, pos, size):
        cls.pos = pos
        cls.size = size

    @classmethod
    def render(cls, pos):
        if not (pos[0] < cls.pos[0] or pos[1] < cls.pos[1] or pos[0] >= cls.pos[0] + cls.size * 28 or pos[1] >= cls.pos[1] + cls.size * 28):
            index = int((pos[0] - cls.pos[0]) / cls.size) + int((pos[1] - cls.pos[1]) / cls.size) * 28
            if index < 784:
                cls.instances[index].update(255)

    @classmethod
    def reset(cls):
        screen.fill((255, 255, 255), (cls.pos[0], cls.pos[1], cls.size * 28, cls.size * 28))
        for i in cls.instances:
            i.color = (0, 0, 0)
            pygame.draw.rect(screen, i.color, (i.pos[0], i.pos[1], i.size, i.size))

    @classmethod
    def return_arr(cls):
        tmp = []
        for i in range(len(cls.instances)):
            tmp.append(cls.instances[i].color[0])
        return numpy.array(tmp)


pixel.init((100, 100), 20)
for i in range(28):
    for j in range(28):
        pixel()
button((200, 700), (100, 100), txt="Reset"), button((400, 700), (100, 100), txt="Predict")

def main():
    while 1:
        pygame.time.Clock().tick(100)
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    pixel.reset()
                elif e.key == pygame.K_ESCAPE:
                    exit(0)
                elif e.key == pygame.K_RETURN:
                    screen.fill((255, 255, 255), (700, 100, 900, 900))
                    img = Image.fromarray(numpy.reshape(pixel.return_arr(), (28, 28)).astype("uint8"))
                    pred = model.predict(numpy.resize(numpy.asarray(img), (1, 28, 28, 1)), verbose=0)
                    pred = sorted(zip(tf.nn.softmax(pred)[0], range(10)), reverse=True)
                    for i in range(len(pred)):
                        screen.blit(pygame.font.Font(pygame.font.get_default_font(), 30).render(str(pred[i][1]) + "  {:.2f}".format(pred[i][0]*100), True, (0, 0, 0)), (700, 100 + 100 * i))
            elif e.type == pygame.QUIT:
                exit(0)
        if pygame.mouse.get_pressed()[0] == 1:
            pos = pygame.mouse.get_pos()
            but = button.get_pressed(pos)
            pixel.render(pos)
            if (but == "Predict"):
                screen.fill((255, 255, 255), (700, 100, 900, 900))
                img = Image.fromarray(numpy.reshape(pixel.return_arr(), (28, 28)).astype("uint8"))
                pred = model.predict(numpy.resize(numpy.asarray(img), (1, 28, 28, 1)), verbose=0)
                pred = sorted(zip(tf.nn.softmax(pred)[0], range(10)), reverse=True)
                for i in range(len(pred)):
                    screen.blit(pygame.font.Font(pygame.font.get_default_font(), 30).render(str(pred[i][1]) + "  {:.2f}".format(pred[i][0]*100), True, (0, 0, 0)), (700, 100 + 100 * i))
            elif (but == "Reset"):
                pixel.reset()
        pygame.display.update()



main()
