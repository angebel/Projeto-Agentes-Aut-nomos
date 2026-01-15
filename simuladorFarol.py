import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from Projeto.ambiente.WorldFarol import WorldFarol
from Projeto.agente.AgentFarol import AgentFarol
from Projeto.agente.EstrategiaNeuralFarol import EstrategiaNeural


class simuladorFarol:
    def __init__(self):
        self.pop_size = 50
        self.archive_threshold = 1.5
        self.k_nearest = 5
        self.archive = []

        self.populacao = []
        for _ in range(self.pop_size):
            strat = EstrategiaNeural(n_inputs=8, n_actions=4)
            self.populacao.append(AgentFarol(strat))

        self.historico_novidade = []
        self.historico_passos = []

    def calcular_novidade(self, agente, vizinhos):
        todos = self.archive + vizinhos
        distancias = []
        ax, ay = agente.posicao_final
        for (vx, vy) in todos:
            d = math.sqrt((ax - vx) ** 2 + (ay - vy) ** 2)
            if d > 0: distancias.append(d)
        distancias.sort()
        if not distancias: return 0
        k = min(len(distancias), self.k_nearest)
        return sum(distancias[:k]) / k

    def treinar(self, total_episodios=2000):
        print(f"--- TREINO DE OTIMIZAÇÃO DESCENDENTE ({total_episodios} Eps) ---")
        start = time.time()


        limite_inicial = 80

        limite_final = 15

        melhor_agente_global = None

        for ep in range(1, total_episodios + 1):
            posicoes_finais = []
            sucessos = []


            progresso = ep / total_episodios
            max_passos_atual = int(limite_inicial - (limite_inicial - limite_final) * progresso)

            random.seed(ep)

            passos_registados = []

            for agente in self.populacao:
                random.seed(ep)
                env = WorldFarol(size=6)
                agente.mundo = env


                agente.visited_cells = set([(0, 0)])
                agente.steps_taken = 0
                agente.last_action = -1
                agente.bates_parede = 0

                chegou = False

                for p in range(max_passos_atual):
                    x_antes, y_antes = env.agentx, env.agenty
                    if agente.agir():
                        chegou = True
                        break
                    if env.agentx == x_antes and env.agenty == y_antes:
                        agente.bates_parede += 1


                if chegou:
                    passos_registados.append(agente.steps_taken)
                    sucessos.append(agente)
                else:
                    passos_registados.append(max_passos_atual)

                agente.posicao_final = (env.agentx, env.agenty)
                posicoes_finais.append(agente.posicao_final)


            self.historico_passos.append(np.mean(passos_registados))


            if len(sucessos) > 0:
                vencedor = sucessos[0]
                if vencedor.steps_taken < 20 and vencedor.bates_parede <= 1:
                    melhor_agente_global = EstrategiaNeural(8, 4)
                    melhor_agente_global.w1 = vencedor.estrategia.w1.copy()
                    melhor_agente_global.b1 = vencedor.estrategia.b1.copy()
                    melhor_agente_global.w2 = vencedor.estrategia.w2.copy()
                    melhor_agente_global.b2 = vencedor.estrategia.b2.copy()


            for agente in self.populacao:
                nov = self.calcular_novidade(agente, posicoes_finais)
                agente.novidade = nov
                if nov > self.archive_threshold:
                    if agente.posicao_final not in self.archive:
                        self.archive.append(agente.posicao_final)

                fitness = len(agente.visited_cells) * 2
                fitness -= agente.bates_parede * 5.0

                if agente.mundo.chegou:

                    fitness += 200 + (max_passos_atual - agente.steps_taken) * 5

                agente.score_final = nov + max(0, fitness)

            if ep % 200 == 0:
                media = np.mean(passos_registados)
                print(f"Ep {ep} (Limite={max_passos_atual}): Média={media:.1f}")


            self.populacao.sort(key=lambda x: x.score_final, reverse=True)
            nova_pop = []
            nova_pop.extend(self.populacao[:5])

            if melhor_agente_global is not None:
                campeao = AgentFarol(melhor_agente_global)
                campeao.estrategia.mutar(power=0.01)
                nova_pop.append(campeao)

            pais = self.populacao[:20]
            while len(nova_pop) < self.pop_size:
                pai = random.choice(pais)
                nova_strat = EstrategiaNeural(8, 4)
                nova_strat.w1 = pai.estrategia.w1.copy()
                nova_strat.b1 = pai.estrategia.b1.copy()
                nova_strat.w2 = pai.estrategia.w2.copy()
                nova_strat.b2 = pai.estrategia.b2.copy()
                nova_strat.mutar(power=0.2)
                nova_pop.append(AgentFarol(nova_strat))

            self.populacao = nova_pop

        print(f"Tempo total: {time.time() - start:.1f}s")
        self.desenhar_graficos_finais()

        if melhor_agente_global:
            self.demonstrar(AgentFarol(melhor_agente_global))

    def desenhar_graficos_finais(self):
        plt.figure(figsize=(10, 6))
        dados = np.array(self.historico_passos)

        window_size = 100
        if len(dados) >= window_size:
            window = np.ones(window_size) / window_size
            dados_suavizados = np.convolve(dados, window, mode='valid')
            x_axis = range(window_size // 2, len(dados) - window_size // 2 + 1)
            plt.plot(dados, color='lightgreen', alpha=0.2, label="Por Episódio")
            plt.plot(x_axis, dados_suavizados, color='darkgreen', linewidth=2, label="Tendência")
        else:
            plt.plot(dados, color='green')

        plt.title("Curva de Aprendizagem ")
        plt.xlabel("Episódios")
        plt.ylabel("Passos Média")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.show(block=False)

    def demonstrar(self, agente):
        print("\n--- DEMONSTRAÇÃO FINAL ---")
        env = WorldFarol(size=6)
        env.force_farol(5, 5)

        agente.mundo = env
        agente.visited_cells = set([(0, 0)])
        agente.steps_taken = 0
        agente.last_action = -1
        agente.bates_parede = 0

        env.render()

        for i in range(40):
            time.sleep(0.3)
            x_ant, y_ant = env.agentx, env.agenty
            venceu = agente.agir()

            if env.agentx == x_ant and env.agenty == y_ant:
                print("!Colisão!")

            env.render()
            if venceu:
                print(f"!!! SUCESSO EM {i + 1} PASSOS !!!")
                break
        plt.show()