from modules.actor import Actor
from modules.actor_cir import Actor as Actor_cir
from modules.critic import Critic

Actor_cir = Actor_cir()
Actor = Actor()
# Actor = Actor(model_path='vggm1-4.npy')
Critic = Critic()